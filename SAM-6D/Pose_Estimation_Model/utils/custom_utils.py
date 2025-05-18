import numpy as np
import cv2


# --- Modularized and improved similarity computation ---
def compute_center_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

# --- Helper for projecting 3D points to image ---
def project_points_to_image(pts_world, cam_R_w2c, cam_t_w2c, K):
    """
    Projects 3D points in world coordinates to 2D image coordinates.
    Returns:
        proj_2d: (N, 2) array of image coordinates (rounded integers)
        cam_z: (N,) array of Z values in camera coordinates
        valid_mask: (N,) boolean array, True if point is in front of camera
    """
    cam_pts = (cam_R_w2c @ pts_world.T + cam_t_w2c).T  # (N, 3)
    cam_z = cam_pts[:, 2]
    valid_mask = cam_z > 0
    cam_pts_valid = cam_pts[valid_mask]
    if cam_pts_valid.shape[0] == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float), valid_mask
    proj_2d = (K @ cam_pts_valid.T).T  # (N_valid, 3)
    proj_2d = proj_2d[:, :2] / proj_2d[:, 2:3]  # (N_valid, 2)
    proj_2d = np.round(proj_2d).astype(int)
    cam_z_valid = cam_z[valid_mask]
    return proj_2d, cam_z_valid, valid_mask

def compute_mask_coverage_ratio(pts_world, cam_R_w2c, cam_t_w2c, K, mask):
    # Project points to image
    proj_2d, _, valid_mask = project_points_to_image(pts_world, cam_R_w2c, cam_t_w2c, K)
    mask = np.squeeze(mask)
    H, W = mask.shape
    if proj_2d.shape[0] == 0:
        return 0.0
    in_bounds = (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) & \
                (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H)
    proj_2d = proj_2d[in_bounds]
    if len(proj_2d) == 0:
        return 0.0
    inside_mask = mask[proj_2d[:, 1], proj_2d[:, 0]] > 0
    return np.sum(inside_mask) / len(proj_2d)

def compute_depth_agreement(pts_world, cam_R_w2c, cam_t_w2c, K, depth_map):
    proj_2d, cam_z_valid, valid_mask = project_points_to_image(pts_world, cam_R_w2c, cam_t_w2c, K)
    H, W = depth_map.shape
    if proj_2d.shape[0] == 0:
        return 0.0

    in_bounds = (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) & \
                (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H)
    if not np.any(in_bounds):
        return 0.0

    proj_2d = proj_2d[in_bounds]
    pred_depth = cam_z_valid[in_bounds]

    proj_x = proj_2d[:, 0].astype(int)
    proj_y = proj_2d[:, 1].astype(int)
    true_depth = depth_map[proj_y, proj_x] / 1000.0

    diff = pred_depth - true_depth
    exact_match = np.abs(diff) < 0.01
    penalized = diff > 0.02
    score = 1.5 * np.sum(exact_match) / len(diff) - 1.5 * np.sum(penalized) / len(diff)
    return score
    
def compute_similarity_score(proposal1, proposal2):
    similarity_score = 0.0
    # Extract transformation info
    R1 = proposal1['cam_R_w2c'].squeeze(0).cpu().numpy()
    t1 = (proposal1['cam_t_w2c'].squeeze(0) / 1000.0).view(3, 1).cpu().numpy()
    R2 = proposal2['cam_R_w2c'].squeeze(0).cpu().numpy()
    t2 = (proposal2['cam_t_w2c'].squeeze(0) / 1000.0).view(3, 1).cpu().numpy()
    K1 = proposal1['cam_K'].squeeze(0).cpu().numpy()
    K2 = proposal2['cam_K'].squeeze(0).cpu().numpy()

    # Convert pts to world coordinates
    R1_inv = R1.T
    t1_inv = -R1_inv @ t1
    R2_inv = R2.T
    t2_inv = -R2_inv @ t2

    pts1 = proposal1['pts'].cpu().numpy()
    pts2 = proposal2['pts'].cpu().numpy()

    pts1_world = (R1_inv @ pts1.T + t1_inv).T
    pts2_world = (R2_inv @ pts2.T + t2_inv).T

    center1 = pts1_world.mean(axis=0)
    center2 = pts2_world.mean(axis=0)

    # Compute metrics
    center_dist = compute_center_distance(center1, center2)
    # similarity_score += -center_dist

    mask1 = proposal1['mask'].cpu().numpy()
    mask2 = proposal2['mask'].cpu().numpy()

    overlap_12 = compute_mask_coverage_ratio(pts1_world, R2, t2, K2, mask2)
    overlap_21 = compute_mask_coverage_ratio(pts2_world, R1, t1, K1, mask1)
    similarity_score += (overlap_12 + overlap_21)

    # depth1 = proposal1['depth'].cpu().numpy()
    # depth2 = proposal2['depth'].cpu().numpy()
    # depth_score_12 = compute_depth_agreement(pts1_world, R2, t2, K2, depth2)
    # depth_score_21 = compute_depth_agreement(pts2_world, R1, t1, K1, depth1)
    # depth_score = 0.5 * (depth_score_12 + depth_score_21)
    # total_similarity = -center_dist + 2.0 * mask_score
    return similarity_score

