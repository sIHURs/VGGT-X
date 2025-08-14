# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import utils.colmap as colmap_utils
from utils.umeyama import umeyama
from utils.metric_torch import evaluate_auc, evaluate_pcd

from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_opt", action="store_true", default=False, help="Use pose optimization for reconstruction")
    parser.add_argument("--save_depth_only", action="store_true", default=False, help="If only save depth")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Number of frames to reconstruct")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--overwrite_pcd", action="store_true", default=False, help="Use fine tracking (slower but more accurate)"
    )
    return parser.parse_args()


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

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    target_scene_dir = os.path.join(f"{os.path.dirname(args.scene_dir)}_vggt_opt", os.path.basename(args.scene_dir))
    os.makedirs(target_scene_dir, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")

    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/images.bin")):
        print("Using order of ground truth images from COLMAP sparse reconstruction")
        images_gt = colmap_utils.read_images_binary(os.path.join(args.scene_dir, "sparse/0/images.bin"))
        images_gt_updated = {id: images_gt[id] for id in list(images_gt.keys())}
        image_path_list = [os.path.join(image_dir, images_gt_updated[id].name) for id in images_gt_updated.keys()]
        base_image_path_list = [os.path.basename(path) for path in image_path_list]
    else:
        images_gt_updated = None
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
        if len(image_path_list) == 0:
            raise ValueError(f"No images found in {image_dir}")
        base_image_path_list = [os.path.basename(path) for path in image_path_list]
    if args.total_frame_num is not None:
        image_path_list = image_path_list[:args.total_frame_num]
        base_image_path_list = base_image_path_list[:args.total_frame_num]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    img_load_resolution = 518

    images, original_coords = load_and_preprocess_images_ratio(image_path_list, img_load_resolution)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype)
    images = images.to(device)

    if args.use_opt:
        import utils.opt as opt_utils
        output = opt_utils.extract_matches(extrinsic, intrinsic, images, base_image_path_list, args.max_query_pts)
        extrinsic, intrinsic = opt_utils.pose_optimization(
            output, extrinsic, intrinsic, images, depth_map, depth_conf,
            base_image_path_list, target_scene_dir=target_scene_dir, shared_intrinsics=args.shared_camera,
        )
    
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # save depth_map and depth_conf as .npy files
    target_depth_dir = os.path.join(target_scene_dir, "estimated_depths")
    target_conf_dir = os.path.join(target_scene_dir, "estimated_confs")
    os.makedirs(target_depth_dir, exist_ok=True)
    os.makedirs(target_conf_dir, exist_ok=True)
    for idx, image_path in tqdm(enumerate(image_path_list), desc="Saving depth maps and confidences"):
        inverse_depth_map = 1 / (depth_map[idx] + 1e-8)  # Avoid division by zero
        normalized_inverse_depth_map = (inverse_depth_map - inverse_depth_map.min()) / (inverse_depth_map.max() - inverse_depth_map.min())
        depth_map_path = os.path.join(target_depth_dir, f"{os.path.basename(image_path)}.npy")
        depth_conf_path = os.path.join(target_conf_dir, f"{os.path.basename(image_path)}.npy")
        np.save(depth_map_path, normalized_inverse_depth_map.squeeze())
        np.save(depth_conf_path, depth_conf[idx].squeeze())
    
    print(f"Saved depth maps and confidences to {target_depth_dir} and {target_conf_dir}")
    if args.save_depth_only:
        return True

    conf_thres_value = 1  # hard-coded to 1 for easier reconstruction
    max_points_for_colmap = 200000  # randomly sample 3D points
    shared_camera = False  # in the feedforward manner, we do not support shared camera
    camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/points3D.bin")):
        pcd_gt = colmap_utils.read_points3D_binary(os.path.join(args.scene_dir, "sparse/0/points3D.bin"))
        # max_points_for_colmap = len(pcd_gt)  # use the number of points in the ground truth as the limit

        if images_gt_updated is not None:
            from utils.umeyama import umeyama

            translation_gt = torch.tensor([image.tvec for image in images_gt_updated.values()], device=device)
            rotation_gt = torch.tensor([colmap_utils.qvec2rotmat(image.qvec) for image in images_gt_updated.values()], device=device)

            # gt w2c
            gt_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
            gt_se3[:, :3, :3] = rotation_gt
            gt_se3[:, 3, :3] = translation_gt

            # pred w2c
            pred_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
            pred_se3[:, :3, :3] = torch.tensor(extrinsic[:, :3, :3], device=device)
            pred_se3[:, 3, :3] = torch.tensor(extrinsic[:, :3, 3], device=device)

            auc_results = evaluate_auc(pred_se3, gt_se3, device)

            # add alignment
            camera_centers_gt = - (gt_se3[:, :3, :3].cpu().numpy().transpose(0, 2, 1) @ gt_se3[:, 3, :3][..., None].cpu().numpy()).squeeze(-1)
            camera_centers_pred = - (pred_se3[:, :3, :3].cpu().numpy().transpose(0, 2, 1) @ pred_se3[:, 3, :3][..., None].cpu().numpy()).squeeze(-1)
            c, R, t = umeyama(camera_centers_gt.T, camera_centers_pred.T)

            # iterate pcd_gt dict and updates xyz of each point
            for point_id, point in pcd_gt.items():
                point.xyz = (c * R @ point.xyz + t.squeeze())
            
            pcd_results = evaluate_pcd(
                pcd_gt, points_3d, depth_conf, images,
                images_gt_updated, original_coords, 
                img_load_resolution, conf_thresh=1.0
            )

            result_file = os.path.join(target_scene_dir, "vggt_results.txt")
            with open(result_file, "w") as f:
                f.write(f"Image Count: {len(images_gt_updated)}\n")
                f.write(f"Relative Rotation Error (degrees): {auc_results['rel_rangle_deg']}\n")
                f.write(f"Relative Translation Error (degrees): {auc_results['rel_tangle_deg']}\n")
                f.write(f"AUC at 30 degrees: {auc_results['Auc_30']}\n")
                f.write(f"Accuracy Mean: {pcd_results['accuracy_mean']}\n")
                f.write(f"Completeness Mean: {pcd_results['completeness_mean']}\n")
                f.write(f"Chamfer Distance: {pcd_results['chamfer_distance']}\n")
            
            points_3d_gt = np.array([point.xyz for point in pcd_gt.values()])
            points_rgb_gt = np.array([point.rgb for point in pcd_gt.values()])
    else:
        print("No ground truth points3D.bin found, using random sampling")
        points_3d_gt = None
        points_rgb_gt = None

    image_size = np.array([depth_map.shape[1], depth_map.shape[2]])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(depth_map.shape[1], depth_map.shape[2]), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= conf_thres_value
    # at most writing max_points_for_colmap 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = (depth_map.shape[2], depth_map.shape[1])

    reconstruction = colmap_utils.rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # first create a folder named f"{args.scene_dir}_vggt", then soft link everything from args.scene_dir except for "sparse"
    for item in os.listdir(args.scene_dir):
        if item != "sparse" and not item.endswith("results.txt"):
            src = os.path.join(args.scene_dir, item)
            dst = os.path.join(target_scene_dir, item)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for file in os.listdir(src):
                    os.symlink(os.path.abspath(os.path.join(src, file)), os.path.abspath(os.path.join(dst, file)))
            else:
                os.symlink(os.path.abspath(src), os.path.abspath(dst))

    print(f"Saving reconstruction to {target_scene_dir}/sparse/0")
    sparse_reconstruction_dir = os.path.join(target_scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(target_scene_dir, "sparse/points.ply"))

    if points_3d_gt is not None and points_rgb_gt is not None and args.overwrite_pcd:
        # Save ground truth point cloud for comparison
        print("Overwriting point cloud with ground truth points")
        colmap_utils.write_points3D_binary(pcd_gt, os.path.join(sparse_reconstruction_dir, "points3D.bin"))
        trimesh.PointCloud(points_3d_gt, colors=points_rgb_gt).export(os.path.join(target_scene_dir, "sparse/points.ply"))

    return True


if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
