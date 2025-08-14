import os
import open3d as o3d
import numpy as np
import torch
import argparse
import utils.colmap as colmap_utils

from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

from evo.core import lie_algebra
from evo.core.trajectory import PoseTrajectory3D
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from utils.umeyama import umeyama
from utils.metric_torch import camera_to_rel_deg, calculate_auc_np

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    device = next(model.parameters()).device
    images = images.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            valid_layers = model.depth_head.intermediate_layer_idx
            if valid_layers[-1] != model.aggregator.aa_block_num - 1:
                valid_layers.append(model.aggregator.aa_block_num - 1)
            aggregated_tokens_list, ps_idx = model.aggregator(images, valid_layers)
            aggregated_tokens_list = [tokens.to(device) if tokens is not None else None for tokens in aggregated_tokens_list]

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT evaluation on given scene")
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing images and colmap results")
    parser.add_argument("--conf_thresh", type=float, default=1, help="Confidence threshold for depth map")
    parser.add_argument("--disable_vis", action="store_true", help="Disable visualization of camera centers")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    sparse_dir = os.path.join(args.data_path, "sparse/0")
    images_dir = os.path.join(args.data_path, "images")

    cameras_gt = colmap_utils.read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images_gt = colmap_utils.read_images_binary(os.path.join(sparse_dir, "images.bin"))
    pcd_gt = colmap_utils.read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))

    images_gt_updated = {id: images_gt[id] for id in list(images_gt.keys())}
    image_path_list = [os.path.join(images_dir, images_gt_updated[id].name) for id in images_gt_updated.keys()]
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load and preprocess images with fixed resolution
    vggt_fixed_resolution = 518
    print(f"[INFO] Loading and preprocessing images with resolution from {args.data_path}")
    images, original_coords = load_and_preprocess_images_ratio(image_path_list, vggt_fixed_resolution)

    # Run VGGT to estimate camera and depth
    torch.cuda.empty_cache()
    print(f"[INFO] Running VGGT to estimate camera and depth on {len(images)} images")
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    print("[INFO] Begin camera pose estimation evaluation")
    fl_gt = torch.tensor([cameras_gt[image.camera_id].params[0:2] for image in images_gt_updated.values()], device=device)
    translation_gt = torch.tensor([image.tvec for image in images_gt_updated.values()], device=device)
    rotation_gt = torch.tensor([colmap_utils.qvec2rotmat(image.qvec) for image in images_gt_updated.values()], device=device)

    # gt w2c
    gt_extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
    gt_extrinsics[:, :3, :3] = rotation_gt
    gt_extrinsics[:, :3, 3] = translation_gt

    # pred w2c
    pred_extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
    pred_extrinsics[:, :3, :3] = torch.tensor(extrinsic[:, :3, :3], device=device)
    pred_extrinsics[:, :3, 3] = torch.tensor(extrinsic[:, :3, 3], device=device)

    poses_gt = np.linalg.inv(gt_extrinsics.cpu().numpy())  # gt c2w
    poses_est = np.linalg.inv(pred_extrinsics.cpu().numpy())  # pred c2w
    frame_ids = list(range(len(image_path_list)))

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)

    alignment_transformation = lie_algebra.sim3(
                *traj_ref.align(traj_est, correct_scale=True, correct_only_scale=False, n=-1))
    # alignment_transformation = traj_ref.align_origin(traj_est) @ alignment_transformation

    gt_extrinsics_aligned =torch.tensor(np.linalg.inv(np.array(traj_ref.poses_se3)), device=device)  # aligned gt w2c

    gt_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
    gt_se3[:, :3, :3] = gt_extrinsics_aligned[:, :3, :3]
    gt_se3[:, 3, :3] = gt_extrinsics_aligned[:, :3, 3]

    pred_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
    pred_se3[:, :3, :3] = pred_extrinsics[:, :3, :3]
    pred_se3[:, 3, :3] = pred_extrinsics[:, :3, 3]

    pcd_xyz_gt_list = []
    pcd_rgb_gt_list = []
    pcd_xyz_sampled_list = []
    pcd_rgb_sampled_list = []

    for k, idx in tqdm(enumerate(list(images_gt_updated.keys()))):

        point3D_ids_gt = images_gt_updated[idx].point3D_ids
        mask_gt = (point3D_ids_gt >= 0) & (images_gt_updated[idx].xys[:, 0] >= 0) & (images_gt_updated[idx].xys[:, 1] >= 0) & \
                    (images_gt_updated[idx].xys[:, 0] < original_coords[k, -2].item()) & (images_gt_updated[idx].xys[:, 1] < original_coords[k, -1].item())

        xys_gt = images_gt_updated[idx].xys[mask_gt]
        pcd_rgb_gt = np.stack([pcd_gt[id].rgb for id in point3D_ids_gt[mask_gt]], axis=0)
        pcd_xyz_gt = np.stack([pcd_gt[id].xyz for id in point3D_ids_gt[mask_gt]], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz_gt)
        distances = np.array(pcd.compute_nearest_neighbor_distance())
        avg_dist = np.mean(distances)
        std_dev_dist = np.std(distances)

        mask_distance = distances < avg_dist
        xys_gt = xys_gt[mask_distance]
        pcd_xyz_gt = pcd_xyz_gt[mask_distance]
        pcd_rgb_gt = pcd_rgb_gt[mask_distance]

        # transform xys_gt to the coordinate on points_3d, which is (N, H, W, 3)
        xys_gt_scaled = np.zeros_like(xys_gt)
        pcd_xyz_sampled = np.zeros_like(pcd_xyz_gt)
        pcd_conf_sampled = np.zeros_like(pcd_rgb_gt[:, 0])  # Assuming confidence is a single channel
        pcd_rgb_sampled = np.zeros_like(pcd_rgb_gt)
        resize_ratio = original_coords[:, -2:].max() / vggt_fixed_resolution

        xys_gt_scaled[:, 0] = xys_gt[:, 0] / resize_ratio + original_coords[k, 0]
        xys_gt_scaled[:, 1] = xys_gt[:, 1] / resize_ratio + original_coords[k, 1]

        xys_gt_scaled[:, 0] = np.clip(xys_gt_scaled[:, 0], 0, points_3d.shape[2] - 1)
        xys_gt_scaled[:, 1] = np.clip(xys_gt_scaled[:, 1], 0, points_3d.shape[1] - 1)
        
        pcd_xyz_sampled = points_3d[k, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)]
        pcd_conf_sampled = depth_conf[k, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)]
        pcd_rgb_sampled = images[k, :, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)].permute(1, 0).cpu().numpy() * 255

        conf_mask = pcd_conf_sampled > args.conf_thresh

        pcd_xyz_gt_list.append(pcd_xyz_gt[conf_mask])
        pcd_xyz_sampled_list.append(pcd_xyz_sampled[conf_mask])
        pcd_rgb_gt_list.append(pcd_rgb_gt[conf_mask])
        pcd_rgb_sampled_list.append(pcd_rgb_sampled[conf_mask])

    pcd_xyz_gt_array = np.concatenate(pcd_xyz_gt_list, axis=0)
    pcd_xyz_sampled_array = np.concatenate(pcd_xyz_sampled_list, axis=0)
    pcd_rgb_gt_array = np.concatenate(pcd_rgb_gt_list, axis=0) / 255.0
    pcd_rgb_sampled_array = np.concatenate(pcd_rgb_sampled_list, axis=0) / 255.0

    pcd_xyz_gt_array = (alignment_transformation[:3, :3] @ pcd_xyz_gt_array.T + alignment_transformation[:3, 3:]).T

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pcd_xyz_gt_array)
    pcd_src.colors = o3d.utility.Vector3dVector(pcd_rgb_gt_array)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(pcd_xyz_sampled_array)
    pcd_tgt.colors = o3d.utility.Vector3dVector(pcd_rgb_sampled_array)

    completeness = pcd_src.compute_point_cloud_distance(pcd_tgt)
    accuracy = pcd_tgt.compute_point_cloud_distance(pcd_src)

    accuracy_mean = np.mean(accuracy)  # to be written to txt file
    completeness_mean = np.mean(completeness)  # to be written to txt file
    chamfer_distance = np.mean(np.concatenate([accuracy, completeness]))  # to be written to txt file

    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_se3, gt_se3, device, 4)
    rError = rel_rangle_deg.cpu().numpy()  # to be written to txt file
    tError = rel_tangle_deg.cpu().numpy()  # to be written to txt file
    Auc_30 = calculate_auc_np(rError, tError, max_threshold=30)  # to be written to txt file

    result_file = os.path.join(args.data_path, "vggt_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Image Count: {len(images_gt_updated)}\n")
        f.write(f"Relative Rotation Error (degrees): {rError.mean()}\n")
        f.write(f"Relative Translation Error (degrees): {tError.mean()}\n")
        f.write(f"AUC at 30 degrees: {Auc_30}\n")
        f.write(f"Accuracy Mean: {accuracy_mean}\n")
        f.write(f"Completeness Mean: {completeness_mean}\n")
        f.write(f"Chamfer Distance: {chamfer_distance}\n")
    
    print(f"[INFO] Results saved to {result_file}")

    if not args.disable_vis:
        # Visualize the camera centers
        print("[INFO] Visualizing camera centers")
        camera_centers_gt = traj_ref.positions_xyz
        camera_centers_pred = traj_est.positions_xyz

        variance_cam = np.var(camera_centers_gt, axis=0)
        ground_plane_indices = np.argsort(variance_cam)[1:]
        print(f"[INFO] Ground plane indices for camera centers: {ground_plane_indices}")

        # Normalize the camera centers to [0, 1]
        # camera_centers_gt -= camera_centers_gt.min(axis=0, keepdims=True)
        # camera_centers_gt /= camera_centers_gt.max(axis=0, keepdims=True)
        # camera_centers_pred -= camera_centers_pred.min(axis=0, keepdims=True)
        # camera_centers_pred /= camera_centers_pred.max(axis=0, keepdims=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure()
        plt.scatter(camera_centers_gt[:, ground_plane_indices[0]], 
                    camera_centers_gt[:, ground_plane_indices[1]], c='blue', label='Camera Centers GT')
        plt.scatter(camera_centers_pred[:, ground_plane_indices[0]], 
                    camera_centers_pred[:, ground_plane_indices[1]], c='red', label='Camera Centers Predicted')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.tight_layout()

        plt.show()
        plt.savefig(os.path.join(args.data_path, "camera_centers.png"))
        print(f"[INFO] Camera centers visualization saved to {os.path.join(args.data_path, 'camera_centers.png')}")

        # Visualize the point cloud
        x_min = max(pcd_xyz_gt_array[:, ground_plane_indices[0]].min(), pcd_xyz_sampled_array[:, ground_plane_indices[0]].min())
        x_max = min(pcd_xyz_gt_array[:, ground_plane_indices[0]].max(), pcd_xyz_sampled_array[:, ground_plane_indices[0]].max())
        y_min = max(pcd_xyz_gt_array[:, ground_plane_indices[1]].min(), pcd_xyz_sampled_array[:, ground_plane_indices[1]].min())
        y_max = min(pcd_xyz_gt_array[:, ground_plane_indices[1]].max(), pcd_xyz_sampled_array[:, ground_plane_indices[1]].max())

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(pcd_xyz_gt_array[:, ground_plane_indices[0]], 
                    pcd_xyz_gt_array[:, ground_plane_indices[1]], c=pcd_rgb_gt_array, s=1)
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Transformed GT Points, {len(image_path_list)} frames")

        plt.subplot(1, 2, 2)
        plt.scatter(pcd_xyz_sampled_array[:, ground_plane_indices[0]], 
                    pcd_xyz_sampled_array[:, ground_plane_indices[1]], c=pcd_rgb_sampled_array, s=1)
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Sampled Points, {len(image_path_list)} frames")
        plt.tight_layout()

        plt.show()
        plt.savefig(os.path.join(args.data_path, "point_cloud.png"))
        print(f"[INFO] Point cloud visualization saved to {os.path.join(args.data_path, 'point_cloud.png')}")

        

