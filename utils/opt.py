import roma
import kornia
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

def make_K_cam_depth(log_focals, pps, trans, quats, min_focals, max_focals, imsizes):
    # make intrinsics
    focals = log_focals.exp().clip(min=min_focals, max=max_focals)
    K = torch.eye(4, dtype=focals.dtype, device=focals.device)[None].expand(len(trans), 4, 4).clone()
    K[:, 0, 0] = K[:, 1, 1] = focals
    K[:, 0:2, 2] = pps * imsizes
    if trans is None:
        return K

    w2cs = torch.eye(4, dtype=trans.dtype, device=trans.device)[None].expand(len(trans), 4, 4).clone()
    w2cs[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(quats, dim=1))
    w2cs[:, :3, 3] = trans

    return K, (w2cs, torch.linalg.inv(w2cs))

def cosine_schedule(alpha, lr_base, lr_end=0):
    lr = lr_end + (lr_base - lr_end) * (1 + np.cos(alpha * np.pi)) / 2
    return lr

def adjust_learning_rate_by_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

def l1_loss(x, y):
    return torch.linalg.norm(x - y, dim=-1)

def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
    if offset is None:
        if gamma == 1:
            return l1_loss
        # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
        offset = (1 / gamma)**(1 / (gamma - 1))

    def loss_func(x, y):
        return (mul * l1_loss(x, y).clip(max=clip) + offset) ** gamma - offset ** gamma
    return loss_func

def image_pair_candidates(extrinsic, pairing_angle_threshold=30, unique_pairs=False):

    pairs, pairs_cnt = {}, 0

    # assert i_map is None or len(i_map) == len(extrinsics)

    num_images = len(extrinsic)
    
    extrinsic_tensor = torch.from_numpy(extrinsic)

    for i in range(num_images):
        
        rot_mat_i = extrinsic_tensor[i:i+1, :3, :3]
        rot_mats_j = extrinsic_tensor[i+1:, :3, :3]

        rot_mat_ij = torch.matmul(rot_mat_i, torch.linalg.inv(rot_mats_j))
        angle_rad = torch.acos((rot_mat_ij.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2)
        angle_deg = angle_rad / np.pi * 180

        i_entry = i
        j_entries = (i + 1 + torch.where(torch.abs(angle_deg) < pairing_angle_threshold)[0]).tolist()

        pairs_cnt += len(j_entries)

        if not i_entry in pairs.keys():
            pairs[i_entry] = []
        pairs[i_entry] = pairs[i_entry] + j_entries

        if not unique_pairs:
            for j_entry in j_entries:
                if not j_entry in pairs.keys():
                    pairs[j_entry] = []
                pairs[j_entry].append(i_entry)

    return pairs, pairs_cnt

@torch.inference_mode()
def extract_matches(extrinsic, intrinsic, images, base_image_path_list, max_query_pts=4096):

    xfeat = torch.hub.load('/home/jing_li/.cache/torch/hub/verlab_accelerated_features_main', 
                           'XFeat', source='local', pretrained=True, top_k=max_query_pts)  # TODO: remove the local path

    pairs, pairs_cnt = image_pair_candidates(extrinsic, pairing_angle_threshold=30, unique_pairs=True)
    print("Total candidate image pairs found: ", pairs_cnt)

    indexes_i = list(range(len(base_image_path_list)-1))  # the last image 
    # indexes_j = [np.random.choice(pairs[idx_i], min(20, len(pairs[idx_i])), replace=False) for idx_i in indexes_i]
    indexes_j = [pairs[idx_i] for idx_i in indexes_i]
    indexes_i = [np.array([idx_i] * len(indexes_j[idx_i])) for idx_i in indexes_i]
    indexes_i = np.concatenate(indexes_i).tolist()
    indexes_j = np.concatenate(indexes_j).tolist()

    batch_size, matches_list = 100, []

    for i in tqdm(range(0, len(indexes_i), batch_size), desc="Matching image pairs..."):
        indexes_i_batch = indexes_i[i:i + batch_size]
        indexes_j_batch = indexes_j[i:i + batch_size]
        
        # Extract features for the batch
        images_i = images[indexes_i_batch]
        images_j = images[indexes_j_batch]
        
        # Match features
        matches_batch = xfeat.match_xfeat_star(images_i, images_j)
        matches_list.extend(matches_batch)

    num_matches = [len(m) for m in matches_list]

    indexes_i_expanded = []
    indexes_j_expanded = []

    for idx, n in enumerate(num_matches):
        indexes_i_expanded.append(np.array([indexes_i[idx]] * n, dtype=np.int64))
        indexes_j_expanded.append(np.array([indexes_j[idx]] * n, dtype=np.int64))
    indexes_i_expanded = np.concatenate(indexes_i_expanded)
    indexes_j_expanded = np.concatenate(indexes_j_expanded)

    image_names_i = np.array(base_image_path_list)[indexes_i_expanded]
    image_names_j = np.array(base_image_path_list)[indexes_j_expanded]

    corr_points_i = torch.cat([matches_list[k][:, :2] for k in range(len(matches_list))], dim=0).cpu()
    corr_points_j = torch.cat([matches_list[k][:, 2:] for k in range(len(matches_list))], dim=0).cpu()

    intrinsic_i = np.zeros((corr_points_i.shape[0], 4, 4), dtype=np.float32)
    intrinsic_j = np.zeros((corr_points_j.shape[0], 4, 4), dtype=np.float32)
    intrinsic_i[:, :3, :3] = intrinsic[indexes_i_expanded]
    intrinsic_j[:, :3, :3] = intrinsic[indexes_j_expanded]
    intrinsic_i[:, 3, 3] = 1.0
    intrinsic_j[:, 3, 3] = 1.0

    extrinsic_i = np.zeros((corr_points_i.shape[0], 4, 4), dtype=np.float32)
    extrinsic_j = np.zeros((corr_points_j.shape[0], 4, 4), dtype=np.float32)
    extrinsic_i[:, :3, :4] = extrinsic[indexes_i_expanded]
    extrinsic_j[:, :3, :4] = extrinsic[indexes_j_expanded]
    extrinsic_i[:, 3, 3] = 1.0
    extrinsic_j[:, 3, 3] = 1.0

    device = corr_points_i.device

    intrinsic_i_tensor = torch.FloatTensor(intrinsic_i).to(device)
    intrinsic_j_tensor = torch.FloatTensor(intrinsic_j).to(device)
    extrinsic_i_tensor = torch.FloatTensor(extrinsic_i).to(device)
    extrinsic_j_tensor = torch.FloatTensor(extrinsic_j).to(device)

    P_i = intrinsic_i_tensor @ extrinsic_i_tensor
    P_j = intrinsic_j_tensor @ extrinsic_j_tensor
    Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
    err = kornia.geometry.symmetrical_epipolar_distance(corr_points_i[:, None, :2], corr_points_j[:, None, :2], Fm, squared=False, eps=1e-08)
    
    hist, bin_edges = torch.histogram(err.cpu(), bins=100, range=(0, 20), density=True)  # move to cpu to avoid CUDA "backend"
    corr_weights = torch.zeros_like(err)
    for i in range(len(bin_edges) - 1):
        mask = (err >= bin_edges[i]) & (err < bin_edges[i + 1])
        if torch.any(mask):
            corr_weights[mask] = (hist[i] * (bin_edges[i + 1] - bin_edges[i])) / (bin_edges[-1] - bin_edges[0])
    corr_weights /= corr_weights.mean()
    
    # set corr_weights to 0 for points outside the image frame
    in_frame_i = (corr_points_i[..., 0] > images.shape[-1]) & (corr_points_i[..., 0] < 0) & \
                    (corr_points_i[..., 1] > images.shape[-2]) & (corr_points_i[..., 1] < 0)
    in_frame_j = (corr_points_j[..., 0] > images.shape[-1]) & (corr_points_j[..., 0] < 0) & \
                    (corr_points_j[..., 1] > images.shape[-2]) & (corr_points_j[..., 1] < 0)
    corr_weights[in_frame_i & in_frame_j] = 0.0
    
    # rearrange corr_points_i_normalized and corr_points_j_normalized to (P, N, 2)
    P, N = len(num_matches), max(num_matches)
    corr_points_i_batched = torch.zeros((P, N, 2), dtype=corr_points_i.dtype, device=corr_points_i.device)
    corr_points_j_batched = torch.zeros((P, N, 2), dtype=corr_points_j.dtype, device=corr_points_j.device)
    corr_weights_batched = torch.zeros((P, N, 1), dtype=corr_weights.dtype, device=corr_weights.device)
    image_names_i_batched = np.zeros((P), dtype=image_names_i.dtype)
    image_names_j_batched = np.zeros((P), dtype=image_names_j.dtype)

    start_idx = 0
    for p in range(P):
        end_idx = start_idx + num_matches[p]
        corr_points_i_batched[p, :num_matches[p]] = corr_points_i[start_idx:end_idx]
        corr_points_j_batched[p, :num_matches[p]] = corr_points_j[start_idx:end_idx]
        corr_weights_batched[p, :num_matches[p]] = corr_weights[start_idx:end_idx]
        image_names_i_batched[p] = image_names_i[start_idx]
        image_names_j_batched[p] = image_names_j[start_idx]
        assert (image_names_i[start_idx:end_idx] == image_names_i_batched[p]).all()
        assert (image_names_j[start_idx:end_idx] == image_names_j_batched[p]).all()
        start_idx = end_idx
    
    output_dict = {
        "corr_points_i": corr_points_i_batched,
        "corr_points_j": corr_points_j_batched,
        "corr_weights": corr_weights_batched,
        "image_names_i": image_names_i_batched,
        "image_names_j": image_names_j_batched,
        "num_matches": num_matches,
    }

    return output_dict

def pose_optimization(match_outputs, 
                      extrinsic, 
                      intrinsic, 
                      images, 
                      depth_map, 
                      depth_conf, 
                      base_image_path_list, 
                      device='cuda',
                      lr_base=5e-4,
                      lr_end=1e-5,
                      niter=300,
                      target_scene_dir=None,
                      shared_intrinsics=True):

    with torch.no_grad():
        imsizes = torch.tensor(images.shape[-2:]).float()
        diags = torch.norm(imsizes)
        min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
        max_focals = 10 * diags

        qvec = roma.rotmat_to_unitquat(torch.tensor(extrinsic[:, :3, :3]))
        tvec = torch.tensor(extrinsic[:, :3, 3])
        log_sizes = torch.zeros(len(qvec))

        pps = torch.tensor(intrinsic[:, :2, 2]) / imsizes[None, :2]  # default principal_point would be (0.5, 0.5)
        base_focals = torch.tensor((intrinsic[:, 0, 0] + intrinsic[:, 1, 1]) / 2)

        # intrinsics parameters
        if shared_intrinsics:
            # Optimize a single set of intrinsics for all cameras. Use averages as init.
            confs = depth_conf.mean(axis=(1, 2))
            weighting = torch.tensor(confs / confs.sum())
            pps = weighting @ pps
            pps = pps.view(1, -1).repeat(len(qvec), 1)
            focal_m = weighting @ base_focals
            log_focals = focal_m.view(1).log().repeat(len(qvec))
        else:
            log_focals = base_focals.log()
        
        depth_map_tensor = torch.tensor(depth_map)  # [B, H, W]

        corr_points_i = match_outputs["corr_points_i"].clone()
        corr_points_j = match_outputs["corr_points_j"].clone()
        corr_weights = match_outputs["corr_weights"].clone()
        num_matches = match_outputs["num_matches"]

        imsizes = imsizes.to(corr_points_i.device)
        depth_map_tensor = depth_map_tensor.to(corr_points_i.device)
        
        corr_points_i_normalized = corr_points_i / imsizes[None, None, [1, 0]] * 2 - 1
        corr_points_j_normalized = corr_points_j / imsizes[None, None, [1, 0]] * 2 - 1

        indexes_i = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_i"]]
        indexes_j = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_j"]]

        depths_i_list, depths_j_list, depth_batch_size = [], [], 16
        for start_idx in range(0, len(corr_points_i_normalized), depth_batch_size):
            end_idx = min(start_idx + depth_batch_size, len(corr_points_i_normalized))
            depths_i_list.append(F.grid_sample(
                depth_map_tensor[indexes_i[start_idx:end_idx]].permute(0, 3, 1, 2),
                corr_points_i_normalized[start_idx:end_idx, None],
                align_corners=True,
                mode='bilinear'
            ).squeeze(1, 2))

            depths_j_list.append(F.grid_sample(
                depth_map_tensor[indexes_j[start_idx:end_idx]].permute(0, 3, 1, 2),
                corr_points_j_normalized[start_idx:end_idx, None],
                align_corners=True,
                mode='bilinear'
            ).squeeze(1, 2))
        
        depths_i = torch.cat(depths_i_list, dim=0).to(device).squeeze(-1)
        depths_j = torch.cat(depths_j_list, dim=0).to(device).squeeze(-1) 

    qvec = qvec.to(device)
    tvec = tvec.to(device)
    log_sizes = log_sizes.to(device)
    min_focals = min_focals.to(device)
    max_focals = max_focals.to(device)
    imsizes = imsizes.to(device)
    pps = pps.to(device)
    log_focals = log_focals.to(device)

    corr_points_i = corr_points_i.to(device)
    corr_points_j = corr_points_j.to(device)
    corr_weight_valid = corr_weights.to(device)
    # corr_weight_valid = corr_weight_valid**(0.5)

    params = [{
        "params": [
            qvec.requires_grad_(True), 
            tvec.requires_grad_(True), 
            log_sizes.requires_grad_(True),
        ],
        "name": ["qvec", "tvec", "log_sizes"]
    }]

    optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))

    loss_list = []
    for iter in tqdm(range(niter or 1), desc="Pose Optimization..."):
        K, (w2cam, cam2w) = make_K_cam_depth(log_focals, pps, tvec, qvec, min_focals, max_focals, imsizes)
        
        
        alpha = (iter / niter)
        lr = cosine_schedule(alpha, lr_base, lr_end)
        adjust_learning_rate_by_lr(optimizer, lr)
        optimizer.zero_grad()

        Ks_i = K[indexes_i]
        Ks_j = K[indexes_j]
        w2cam_i = w2cam[indexes_i]
        w2cam_j = w2cam[indexes_j]
        cam2w_i = cam2w[indexes_i]
        cam2w_j = cam2w[indexes_j]

        loss = 0.0

        sizes = log_sizes.exp()
        global_scaling = 1 / sizes.min()
        depths_i_scaled = depths_i * global_scaling * sizes[indexes_i, None]
        depths_j_scaled = depths_j * global_scaling * sizes[indexes_j, None]
        
        cam_coords_i = torch.stack([
            (corr_points_i[..., 0] - Ks_i[:, None, 0, 2]) / Ks_i[:, None, 0, 0],
            (corr_points_i[..., 1] - Ks_i[:, None, 1, 2]) / Ks_i[:, None, 1, 1],
            depths_i_scaled
        ], dim=-1)
        cam_coords_j = torch.stack([
            (corr_points_j[..., 0] - Ks_j[:, None, 0, 2]) / Ks_j[:, None, 0, 0],
            (corr_points_j[..., 1] - Ks_j[:, None, 1, 2]) / Ks_j[:, None, 1, 1],
            depths_j_scaled
        ], dim=-1)
        world_coords_i = (cam2w_i[:, :3, :3] @ cam_coords_i.permute(0, 2, 1)).permute(0, 2, 1) + cam2w_i[:, None, :3, 3]
        world_coords_j = (cam2w_j[:, :3, :3] @ cam_coords_j.permute(0, 2, 1)).permute(0, 2, 1) + cam2w_j[:, None, :3, 3]
        
        loss = ((world_coords_i - world_coords_j).abs() * corr_weight_valid).mean() * 0.05

        P_i = Ks_i @ w2cam_i
        P_j = Ks_j @ w2cam_j
        Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
        err = kornia.geometry.symmetrical_epipolar_distance(corr_points_i, corr_points_j, Fm, squared=False, eps=1e-08)
        loss = loss + (err * corr_weight_valid.squeeze(-1)).mean() * 1.0
        
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
    
    if target_scene_dir is not None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title(f'Loss Curve, final loss={loss_list[-1]:.4f}')
        plt.show()
        plt.savefig(f"{target_scene_dir}/loss_curve_pose_opt.png")

    
    output_extrinsic = w2cam[:, :3, :4].detach().cpu().numpy()
    output_intrinsic = K[:, :3, :3].detach().cpu().numpy()

    return output_extrinsic, output_intrinsic