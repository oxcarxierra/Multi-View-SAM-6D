import gorilla
from tqdm import tqdm
import argparse
import os
import sys
import os.path as osp
import time
import logging
import numpy as np
import random
import importlib
import pickle as cPickle
import json
import torch
from PIL import Image
from utils.custom_utils import compute_similarity_score, remove_outliers_lof
from sklearn.cluster import AffinityPropagation
from collections import defaultdict
from torchcpd import RigidRegistration
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
# from draw_utils import draw_detections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
MAX_PROPOSALS = 10
BATCH_LIMIT = 4  # adjust if needed

cluster_outputs = {
    'pred_pose_score': [],
    'pred_R': [],
    'pred_t': []
}


detetion_paths = {
    'ycbv': '../Instance_Segmentation_Model/log/sam/result_ycbv.json',
    'tudl': '../Instance_Segmentation_Model/log/sam/result_tudl.json',
    'tless': '../Instance_Segmentation_Model/log/sam/result_tless_scene_01-10.json',
    'lmo': '../Instance_Segmentation_Model/log/sam/result_lmo.json',
    'itodd': '../Instance_Segmentation_Model/log/sam/result_itodd.json',
    'icbin': '../Instance_Segmentation_Model/log/sam/result_icbin.json',
    'hb': '../Instance_Segmentation_Model/log/sam/result_hb.json'
}

def print_gpu_memory(stage=""):
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="index of gpu")
    parser.add_argument("--model",
                        type=str,
                        default="multiview_pose_estimation_model",
                        help="name of model")
    parser.add_argument("--config",
                        type=str,
                        default="config/base_multiview.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="tless",
                        help="")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="checkpoints/sam-6d-pem-base.pth",
                        help="path to checkpoint file")
    parser.add_argument("--iter",
                        type=int,
                        default=0,
                        help="iter num. for testing")
    parser.add_argument("--view",
                        type=int,
                        default=21,
                        help="view number of templates")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="experiment id")
    parser.add_argument("--visualization",
                        type=bool,
                        default=True,
                        help="visualize PEM results")
    parser.add_argument("--n_multiview",
                        type=int,
                        default=5,
                        help="number of multiview images for each scene")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_path = args.checkpoint_path
    cfg.test_iter = args.iter
    cfg.dataset = args.dataset
    cfg.visualization = args.visualization
    cfg.test_dataset.n_multiview = args.n_multiview

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    return cfg

def compute_bbox_iou(pts1, pts2):
    min1, max1 = pts1.min(dim=0)[0], pts1.max(dim=0)[0]
    min2, max2 = pts2.min(dim=0)[0], pts2.max(dim=0)[0]
    
    inter_min = torch.max(min1, min2)
    inter_max = torch.min(max1, max2)
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    inter_vol = inter_dims.prod().item()

    vol1 = (max1 - min1).prod().item()
    vol2 = (max2 - min2).prod().item()

    union_vol = vol1 + vol2 - inter_vol
    if union_vol == 0:
        return 0.0
    return inter_vol / union_vol


# def print_gpu_memory(label=""):
#     allocated = torch.cuda.memory_allocated() / 1024**2
#     reserved = torch.cuda.memory_reserved() / 1024**2
#     print(f"[GPU:{label}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def resample_pointcloud(pts, target_size=2048):
    N = pts.shape[0]
    if N == target_size:
        return pts, torch.arange(N)
    elif N > target_size:
        idx = torch.randperm(N)[:target_size]
        return pts[idx], idx
    else:
        pad_count = target_size - N
        pad_idx = torch.randint(0, N, (pad_count,))
        padded = torch.cat([pts, pts[pad_idx]], dim=0)
        idx = torch.cat([torch.arange(N), pad_idx])
        return [padded, idx]


def clean_pointcloud_torch(
    pts_tensor,
    sor_on=True,
    dbscan_on=True,
    sor_k=20,
    sor_std_ratio=2.0,
    dbscan_eps=0.02,
    dbscan_min_samples=10
):
    """
    Lightweight point cloud cleanup using PyTorch and scikit-learn.
    Steps:
    - Statistical outlier removal (SOR)
    - DBSCAN clustering to isolate largest connected component
    """

    orig_pts = pts_tensor
    pts = pts_tensor.clone()

    # --------- Statistical Outlier Removal ---------
    if sor_on and pts.shape[0] > sor_k:
        print("[CLEANUP] Running SOR")
        dists = torch.cdist(pts, pts, p=2)
        knn_dists, _ = dists.topk(sor_k + 1, largest=False)  # +1 for self
        knn_dists = knn_dists[:, 1:]  # exclude self-distance
        mean_dists = knn_dists.mean(dim=1)

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + sor_std_ratio * global_std

        mask = mean_dists < threshold
        pts = pts[mask]
        print(f"[CLEANUP] SOR: {orig_pts.shape[0]} → {pts.shape[0]} points")

        if pts.shape[0] < 50:
            print("[WARNING] Too few points left after SOR — skipping cleanup")
            return orig_pts

    # --------- DBSCAN Clustering ---------
    if dbscan_on and pts.shape[0] >= dbscan_min_samples:
        print("[CLEANUP] Running DBSCAN")
        pts_np = pts.cpu().numpy()
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(pts_np)
        labels = db.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # remove noise label

        if len(unique_labels) == 0:
            print("[WARNING] No clusters found — skipping DBSCAN")
            return pts

        # Keep only the largest cluster
        counts = np.bincount(labels[labels >= 0])
        main_label = np.argmax(counts)
        keep_indices = np.where(labels == main_label)[0]
        pts = pts[keep_indices]
        print(f"[CLEANUP] DBSCAN kept {len(pts)} points from cluster {main_label}")

        if pts.shape[0] < 50:
            print("[WARNING] Too few points after DBSCAN — falling back")
            return orig_pts

    return pts


# def run_icp_refinement(source_pts, target_pts, target_normals=None, threshold=0.01): # -> with open3d, heavy (1.1GB) and overkill for our use case
#     """
#     Runs point-to-plane ICP to refine pose.
    
#     Args:
#         source_pts (np.ndarray): Nx3 model point cloud.
#         target_pts (np.ndarray): Nx3 observed point cloud.
#         target_normals (np.ndarray): Nx3 normals (optional).
#         threshold (float): Distance threshold for ICP.
        
#     Returns:
#         refined_trans (4x4 np.ndarray): Transformation matrix.
#     """
#     source = o3d.geometry.PointCloud()
#     source.points = o3d.utility.Vector3dVector(source_pts)
    
#     target = o3d.geometry.PointCloud()
#     target.points = o3d.utility.Vector3dVector(target_pts)
    
#     if target_normals is not None:
#         target.normals = o3d.utility.Vector3dVector(target_normals)
#     else:
#         target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    
#     result_icp = o3d.pipelines.registration.registration_icp(
#         source, target, threshold,
#         np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )
#     return result_icp.transformation

def run_icp_refinement(source_pts, target_pts, threshold=0.01, max_iter=50): # using torchcpd instead of open3d, but doing point-to-point ICP instead of point-to-plane
    """
    ICP with torchcpd (point-to-point), works fully in PyTorch.
    Args:
        source_pts: (N, 3) numpy array
        target_pts: (N, 3) numpy array
    Returns:
        4x4 transformation matrix as np.ndarray
    """
    print("[ICP] Running ICP refinement")

    # Ensure both are numpy float32
    if isinstance(source_pts, torch.Tensor):
        source_pts = source_pts.detach().cpu().numpy()
    if isinstance(target_pts, torch.Tensor):
        target_pts = target_pts.detach().cpu().numpy()
    
    source_pts = source_pts.astype(np.float32)
    target_pts = target_pts.astype(np.float32)

    assert isinstance(source_pts, np.ndarray), f"Expected np.ndarray, got {type(source_pts)}"
    assert isinstance(target_pts, np.ndarray), f"Expected np.ndarray, got {type(target_pts)}"

    reg = RigidRegistration(X=target_pts, Y=source_pts, max_iterations=max_iter)
    result = reg.register()
    print(f"[DEBUG] ICP return type: {type(result)}")
    print(f"[DEBUG] ICP return content: {result}")

    try:
        TY, (sigma2, R, t) = reg.register()
    except ValueError as e:
        print("[ERROR] Unpacking failed:", e)
        R, t = reg.register()  # fallback in case it returns just (R, t)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.cpu().numpy()
    T[:3, 3] = t.cpu().numpy().flatten()
    return T


def test(model, cfg, save_path, dataset_name, detetion_path):
    model.eval()
    bs = cfg.test_dataloader.bs
    cluster_summary_lines = []

    # build dataloader
    def identity_collate(batch):
        return batch[0]

    dataset = importlib.import_module(cfg.test_dataset.name)

    # print("Imported dataset module:", dataset)
    # print("Available attributes:", dir(dataset))
    print("Dataset class:", dataset.BOPMultiviewTestset)
    dataset = dataset.BOPMultiviewTestset(cfg.test_dataset, dataset_name, detetion_path)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False, # cfg.test_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.test_dataloader.drop_last,
            pin_memory=cfg.test_dataloader.pin_memory,
            collate_fn=identity_collate
        )

    # prepare for target objects
    all_tem, all_tem_pts, all_tem_choose = dataset.get_templates()

    with torch.no_grad():
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    lines = []
    cluster_summary_lines = []
    cluster_summary_lines.append("scene_id,img_id,obj_ids\n")
    cluster_csv_path = os.path.join(os.path.dirname(save_path), "cluster_result.csv")

    with tqdm(total=len(dataloder)) as t:
        print("Total number of batches:", len(dataloder))
        for group_idx, data in enumerate(dataloder):
            print(f"=====================> Processing {group_idx+1}th multiview batch")
            # data = dict_keys(['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'scene_id', 'img_id', 'seg_time']) -> for same scene & image show all proposals
            torch.cuda.synchronize()
            end = time.time()

            assert len(data) == cfg.test_dataset.n_multiview, \
                f"Expected {cfg.test_dataset.n_multiview} images, but got {len(data)} images."

            multiview_datas = []
            for single_img_data in data:
                single_image_proposals = []
                for proposal_idx in range(len(single_img_data['obj'])):
                    proposal = {}
                    for key in single_img_data.keys():
                        if key in ['img_id', 'scene_id', 'seg_time']:
                            proposal[key] = single_img_data[key][0].cuda()
                        else:
                            proposal[key] = single_img_data[key][proposal_idx].cuda()

                    single_image_proposals.append(proposal)
                multiview_datas.append(single_image_proposals)
            all_proposals = sum(multiview_datas, [])  # flatten
            
            ## Step 1: Cluster proposals
            # Compute score and make similarity matrix
            N = len(all_proposals)
            S = np.zeros((N, N))

            for i in range(N):
                for j in range(i + 1, N):
                    similarity_score = compute_similarity_score(all_proposals[i], all_proposals[j])
                    S[i, j] = S[j, i] = similarity_score

            np.fill_diagonal(S, np.median(S))
            # Run Affinity Propagation clustering
            clustering = AffinityPropagation(affinity='precomputed', random_state=0)
            labels = clustering.fit_predict(S)

            # Group proposals by cluster label
            cluster_proposals = defaultdict(list)
            for proposal, label in zip(all_proposals, labels):
                cluster_proposals[label].append(proposal)

            # Print cluster summary
            print(f"[Clustering] Estimated number of clusters (raw): {len(cluster_proposals)}")

            to_remove = [label for label, proposals in cluster_proposals.items() if len(proposals) < 2]
            for label in to_remove:
                cluster_proposals.pop(label)

            print(f"[Clustering] Estimated number of clusters (filtered): {len(cluster_proposals)}")
            for label, proposals in cluster_proposals.items():
                print(f"\n[Cluster {label}] contains {len(proposals)} proposals:")
                for proposal in proposals:
                    obj_id = proposal['obj_id'].item()
                    score = proposal['score'].item()
                    print(f"  obj_id: {obj_id}, score: {score:.4f}")

            ## Step 2 : Inference multiview pose estimation for each objects with best ID
            cluster_results = {}
            obj_ids = []
            for label, proposals_for_obj in cluster_proposals.items():

                merged_pts_list = []
                merged_dense_fo_list = []

                for proposal in proposals_for_obj:
                    pts = proposal['pts']
                    R_w2c = proposal['cam_R_w2c'].squeeze(0)
                    t_w2c = proposal['cam_t_w2c'].squeeze(0) / 1000.0
                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c.view(3, 1)
                    pts_world = (R_c2w @ pts.T + t_c2w).T
                    merged_pts_list.append(pts_world)

                if not merged_pts_list:
                    continue

                # ------------------------
                # Merge to single pointcloud
                # ------------------------
                all_pts = torch.cat(merged_pts_list, dim=0)
                filtered_pts = remove_outliers_lof(all_pts, n_neighbors=20, contamination=0.01)
                n_points = filtered_pts.shape[0]
                sampled_idx = torch.randperm(n_points)[:proposals_for_obj[0]['pts'].shape[0]]
                merged_pts = filtered_pts[sampled_idx]  # shape: (2048, 3)

                # # ------------------------
                
                # # # --- 1. Estimate adaptive parameters --- first try
                # # bbox = all_pts.max(dim=0)[0] - all_pts.min(dim=0)[0]
                # # scale = bbox.norm().item()
                # # eps = max(0.005, scale * 0.05)  # in meters

                # # N = all_pts.shape[0]
                # # min_samples = max(10, int(np.sqrt(N)))  # or try int(0.01 * N)

                # # # ------------------------
                # # # Run DBSCAN on merged cloud (filter noise, keep main object)
                # # # ------------------------
                # # pts_np = all_pts.cpu().numpy()
                # # labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_np).labels_

                # # valid_labels = [l for l in set(labels) if l != -1]
                # # if valid_labels:
                # #     counts = np.bincount(labels[labels != -1])
                # #     main_label = valid_labels[np.argmax(counts)]
                # #     keep_indices = np.where(labels == main_label)[0]
                # #     all_pts = all_pts[keep_indices]
                # #     print(f"[DBSCAN] Adaptive: eps={eps:.4f}, min_samples={min_samples}: {len(pts_np)} → {len(keep_indices)} points kept")
                # # else:
                # #     print(f"[DBSCAN] No valid cluster found — using all {N} points as fallback")
                # # # end first try


                # # #vME
                # # # ---------------------
                # # # HDBSCAN (via PCA whitening)
                # # # ---------------------
                # # pts_np = all_pts.cpu().numpy()
                # # N = pts_np.shape[0]

                # # # 1. PCA whitening: decorrelate + normalize axes
                # # pca = PCA(whiten=True)
                # # pts_whitened = pca.fit_transform(pts_np)

                # # # 2. Adaptive clustering parameters
                # # bbox = all_pts.max(dim=0)[0] - all_pts.min(dim=0)[0]
                # # scale = bbox.norm().item()
                # # # eps = max(0.005, scale * 0.05)
                # # min_samples = max(10, 0.001 * N)# int(np.sqrt(N)//3))
                # # min_cluster_size = 0.8*N #2*N//3

                # # # 3. Run DBSCAN in whitened space
                # # # labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_whitened).labels_
                # # labels = HDBSCAN(min_cluster_size=min_samples, min_samples=10).fit(pts_whitened).labels_

                # # valid_labels = [l for l in set(labels) if l != -1]
                # # if valid_labels:
                # #     counts = np.bincount(labels[labels != -1])
                # #     main_label = valid_labels[np.argmax(counts)]
                # #     keep_indices = np.where(labels == main_label)[0]
                # #     all_pts = all_pts[keep_indices]
                # #     print(f"[HDBSCAN] min_cluster_size={min_cluster_size}, min_samples={min_samples}, kept {len(keep_indices)}/{N}")
                # # else:
                # #     print(f"[HDBSCAN] No cluster found — using all {N} points as fallback")
                # # # ------------------------
                    
                # # Optional: add light SOR here if you want extra filtering
                # cleaned_pts = clean_pointcloud_torch(all_pts, sor_on=True, dbscan_on=True) # dbscan_on=False before

                # # ------------------------
                # # Resample to fixed number of points for PEM
                # # ------------------------
                # merged_pts, _ = resample_pointcloud(all_pts, target_size=proposals_for_obj[0]['pts'].shape[0])
                # merged_dense_fo = all_dense_fo # cleaned_dense_fo[idx]
                # # ------------------------ # end vME
                # # ------------------------
                cluster_obj_id = max(proposals_for_obj, key=lambda p: p['score'].item())['obj_id'].item()
                model_for_cluster_obj_id = next(proposal['model'] for proposal in proposals_for_obj if proposal['obj_id'].item() == cluster_obj_id)
                obj_ids.append(cluster_obj_id)

                # === PEM Inference with Memory-Safe Batching ===
                BATCH_LIMIT = 4

                cluster_outputs = {
                    'pred_pose_score': [],
                    'pred_R': [],
                    'pred_t': [],
                    'score': []
                }
                used_proposals = []

                N = len(proposals_for_obj)

                for i in range(0, N, BATCH_LIMIT):
                    # print_gpu_memory(f"Before PEM batch {i//BATCH_LIMIT + 1}")
                    batch_proposals = proposals_for_obj[i:i + BATCH_LIMIT]
                    used_proposals.extend(batch_proposals)

                    batch_inputs = {
                        'pts': [],
                        'rgb': [],
                        'rgb_choose': [],
                        'model': [],
                        'dense_po': [],
                        'dense_fo': [],
                        'K': [],
                        'score': [],
                        'R_w2c': [],
                        't_w2c': []
                    }

                    for proposal in batch_proposals:

                        obj = proposal['obj'][0].item()
                        # merged_pts_camera_frame = (R_w2c @ merged_pts.T + t_w2c.view(3, 1)).T
                        
                        # save_dir = f"/home/obuset/SAM-6D/SAM-6D/Pose_Estimation_Model/log/intermediate_results_DBSCAN_ICP_P2P_corrected/intermediate_results_scene_{proposal['scene_id'].item():02d}_v_DBSCAN_ICP_P2P_corrected/"
                        # os.makedirs(save_dir, exist_ok=True)
                        # np.save(.join(save_dir, f"img{(proposal['img_id'].item()):02d}_cluster{proposal['obj_id'].item():06d}.npy"), merged_pts_camera_frame.cpu().numpy())
                        # print(f"Saved merged points for cluster {proposal['obj_id'].item()} in scene {proposal['scene_id'].item()} and image {proposal['img_id'].item()}")

                        batch_inputs['pts'].append(proposal['pts'].unsqueeze(0))
                        batch_inputs['rgb'].append(proposal['rgb'].unsqueeze(0))
                        batch_inputs['rgb_choose'].append(proposal['rgb_choose'].unsqueeze(0))
                        batch_inputs['model'].append(proposal['model'].unsqueeze(0))
                        # batch_inputs['model'].append(model_for_cluster_obj_id.unsqueeze(0))
                        batch_inputs['dense_po'].append(dense_po[obj].unsqueeze(0))
                        batch_inputs['dense_fo'].append(dense_fo[obj].unsqueeze(0))
                        batch_inputs['K'].append(proposal['cam_K'].unsqueeze(0))
                        batch_inputs['score'].append(proposal['score'].view(1))
                        batch_inputs['R_w2c'].append(proposal['cam_R_w2c'].unsqueeze(0))
                        batch_inputs['t_w2c'].append(proposal['cam_t_w2c'].unsqueeze(0))

                    # Concatenate and send to CUDA
                    for key in batch_inputs:
                        batch_inputs[key] = torch.cat(batch_inputs[key], dim=0).cuda()

                    with torch.no_grad():
                        sub_output = model(batch_inputs)
                    # print_gpu_memory(f"After PEM batch {i//BATCH_LIMIT + 1}")

                    for k in cluster_outputs:
                        cluster_outputs[k].append(sub_output[k].cpu())  # keep on CPU

                    del batch_inputs, sub_output
                    torch.cuda.empty_cache()

                    # print_gpu_memory("After clearing cache")

                # === Reassemble PEM outputs and find best
                end_points = {k: torch.cat(v, dim=0) for k, v in cluster_outputs.items()}

                # Debug check
                assert len(used_proposals) == end_points['pred_pose_score'].shape[0], \
                    f"Mismatched proposal/output sizes: {len(used_proposals)} vs {end_points['pred_pose_score'].shape[0]}"

                # Convert only once, store result
                pred_scores = end_points['pred_pose_score'].cuda() * end_points['score'].cuda()

                print(f"[DEBUG] #used_proposals: {len(used_proposals)}")
                print(f"[DEBUG] #pred_scores: {end_points['pred_pose_score'].shape[0]}")


                best_idx = torch.argmax(pred_scores).item()
                # best_proposal = proposals_for_obj[best_idx]
                best_proposal = used_proposals[best_idx]
                best_proposal_obj_id = best_proposal['obj_id'].item()   

                # # Get pred pose on GPU
                # R_pred = end_points['pred_R'][best_idx].cuda()
                # t_pred = end_points['pred_t'][best_idx].view(3, 1).cuda()

                # Camera-to-world transform
                R_c2w = best_proposal['cam_R_w2c'].squeeze(0).T.cuda()
                t_c2w = -R_c2w @ (best_proposal['cam_t_w2c'].squeeze(0) / 1000.0).view(3, 1).cuda()
                R_pred = R_c2w @ end_points['pred_R'].cuda()[best_idx]
                t_pred = (R_c2w @ end_points['pred_t'].cuda()[best_idx].view(3, 1) + t_c2w).view(-1)

                # ================ Merging Points and Applying Pose =================
                # # Apply transformation
                # R_final = R_c2w @ R_pred
                # t_final = (R_c2w @ t_pred + t_c2w).view(-1)

                # # Save
                # cluster_results[label] = {
                #     'obj_id': best_proposal_obj_id,
                #     'merged_points': merged_pts,
                #     'pem_score': pred_scores[best_idx].item(),
                #     'R': R_final,
                #     't': t_final
                # }
                # ================ ICP Refinement ================
                # Convert model points to camera frame using SAM-6D prediction
                model_pts = model_for_cluster_obj_id.cpu().numpy().astype(np.float32)        # shape: (N, 3)
                T_init = np.eye(4, dtype=np.float32)
                T_init[:3, :3] = R_pred.cpu().numpy().astype(np.float32)
                T_init[:3, 3] = t_pred.view(3).cpu().numpy().astype(np.float32)
                model_pts_transformed = (T_init[:3, :3] @ model_pts.T).T + T_init[:3, 3]
                assert isinstance(model_pts_transformed, np.ndarray), "model_pts_transformed is not numpy"


                # Target: merged observation cloud (torch → numpy)
                merged_pts_camera_frame = (
                    best_proposal['cam_R_w2c'].squeeze(0) @ merged_pts.T +
                    best_proposal['cam_t_w2c'].squeeze(0).view(3, 1) / 1000.0
                ).T
                target_pts = merged_pts_camera_frame.cpu().numpy().astype(np.float32)
                assert isinstance(target_pts, np.ndarray), "target_pts is not numpy"


                # # Target = observed point cloud in camera frame
                # merged_pts_camera_frame = (best_proposal['cam_R_w2c'].squeeze(0) @ merged_pts.T + best_proposal['cam_t_w2c'].squeeze(0).view(3, 1) / 1000.0).T # shape: (2048, 3), recall merged_pts is already in world frame
                # target_pts = merged_pts_camera_frame.cpu().numpy()

                # if isinstance(model_pts_transformed, torch.Tensor):
                #     model_pts_transformed = model_pts_transformed.cpu().numpy()
                # else:
                #     model_pts_transformed = torch.from_numpy(model_pts.astype(np.float32))
                #     R_init = torch.from_numpy(T_init[:3, :3])  # already float32
                #     t_init = torch.from_numpy(T_init[:3, 3])   # already float32

                #     model_pts_transformed = (R_init @ model_pts_transformed.T).T + t_init
                #     model_pts_transformed = model_pts_transformed.numpy()

                # Run ICP
                T_icp = run_icp_refinement(model_pts_transformed, target_pts, threshold=0.01)

                # Apply ICP to the initial pose
                T_refined = T_icp @ T_init
                R_refined = T_refined[:3, :3]
                t_refined = T_refined[:3, 3]

                # Transform refined pose to world frame
                # R_final = R_c2w @ torch.from_numpy(R_refined).cuda()
                # t_final = (R_c2w @ torch.from_numpy(t_refined).view(3, 1).cuda() + t_c2w).view(-1)

                # Fix double vs float issue
                R_final = R_c2w @ torch.from_numpy(R_refined.astype(np.float32)).cuda()
                t_final = (R_c2w @ torch.from_numpy(t_refined.astype(np.float32)).view(3, 1).cuda() + t_c2w).view(-1)

                # Save
                cluster_results[label] = {
                    'obj_id': best_proposal_obj_id,
                    'merged_points': merged_pts,
                    'pem_score': pred_scores[best_idx].item(),
                    'R': R_final,
                    't': t_final
                }


                print(f"\n[Cluster {label}] Best Proposal: obj_id: {best_proposal_obj_id}, score: {pred_scores[best_idx].item():.4f}")
                for proposal_idx in range(len(proposals_for_obj)):
                    print(f"Proposal {proposal_idx}: obj_id: {proposals_for_obj[proposal_idx]['obj_id'].item()}, score: {pred_scores[proposal_idx]:.4f}")
                
            
            cluster_summary_lines.append(f"{all_proposals[0]['scene_id'].view(-1)[0].item()},{group_idx+1},\"{sorted(obj_ids)}\"\n")
            print(f"Cluster summary for scene {all_proposals[0]['scene_id'].item()}, group {group_idx+1}: {sorted(obj_ids)}")
            with open(cluster_csv_path, 'w+') as f:
                f.writelines(cluster_summary_lines)
            # Step 3:  Write final pose of each cluster to every image in multiview_datas (sorted by img_id)
            sorted_views = sorted(multiview_datas, key=lambda x: x[0]['img_id'].item())

            for view_proposals in sorted_views:
                proposal = view_proposals[0]
                R_w2c = proposal['cam_R_w2c'].squeeze(0)
                t_w2c = proposal['cam_t_w2c'].squeeze(0) / 1000.0
                scene_id = proposal['scene_id'].item()
                img_id = proposal['img_id'].item()

                for label, result in cluster_results.items():
                    obj_id = result['obj_id']
                    R_final = result['R']
                    t_final = result['t']
                    pem_score = result['pem_score']
                    merged_points = result['merged_points']
                    # Calculated earlier as 
                    # merged_pts_camera_frame = (best_proposal['cam_R_w2c'].squeeze(0) @ merged_pts.T + best_proposal['cam_t_w2c'].squeeze(0).view(3, 1) / 1000.0).T # shape: (2048, 3), recall merged_pts is already in world frame

                    merged_pts_camera_frame = (R_w2c @ merged_points.T + t_w2c.view(3, 1)).T

                    R_img = R_w2c @ R_final
                    t_img = ((R_w2c @ t_final.view(3, 1) + t_w2c.view(3, 1)).view(-1))*1000

                    image_time = 0
                    line = ','.join((
                        str(scene_id),
                        str(img_id),
                        str(obj_id),
                        f'{pem_score:.6f}',
                        ' '.join(f'{v:.8f}' for v in R_img.flatten().tolist()),
                        ' '.join(f'{v:.8f}' for v in t_img.tolist()),
                        str(image_time)
                    )) + '\n'
                    lines.append(line)

                    with open(save_path, 'w+') as f:
                        f.writelines(lines)
                    
                    if cfg.visualization and scene_id in [1]:
                        # print(f"Visualizing scene {scene_id}, image {img_id}, cluster {label}, object {obj_id}")
                        vis_dir = save_path.replace(f'result_tless_scene_01-10_v_DBSCAN_ICP_P2P_corrected.csv',f"pem_visualization/scene_{scene_id:02d}_v_DBSCAN_ICP_P2P_corrected/")
                        os.makedirs(os.path.dirname(vis_dir), exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"group{group_idx+1}_img{img_id}_cluster{label}_obj{obj_id}.png")
                        img_path = BASE_DIR + "/../Data/BOP/tless/test/" + str(scene_id).zfill(6) + "/rgb/" + str(img_id).zfill(6) + ".png"
                        cad_path = BASE_DIR + "/../Data/BOP/tless/models_cad/obj_" + str(obj_id).zfill(6) + ".ply"
                        img = np.array(Image.open(img_path))
                        import trimesh
                        mesh = trimesh.load_mesh(cad_path)
                        model_points = mesh.sample(512).astype(np.float32)
                        K_ = proposal['cam_K'].cpu().numpy()
                        vis_img = visualize(img, [merged_pts_camera_frame.cpu().numpy()], [R_img.cpu().numpy()], [t_img.cpu().numpy()], model_points, K_, vis_path)
            t.update(1)
                    
            # Clear data structures
            cluster_proposals.clear()
            multiview_datas.clear()

    with open(save_path, 'w+') as f:
        f.writelines(lines)
    # cluster_csv_path = os.path.join(os.path.dirname(save_path), "cluster_result.csv")
    # with open(cluster_csv_path, 'w+') as f:
    #     f.write("scene_id,img_id,num_clusters\n")
    #     f.writelines(cluster_summary_lines)

def visualize(rgb, pointcloud, pred_rot, pred_trans, model_points, K, save_path):
    from draw_utils import draw_detections
    img = draw_detections(rgb, pointcloud, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    # print(f"Visualization saved to {save_path}")
    # return img
    prediction = Image.open(save_path)

    rgb_pil = Image.fromarray(np.uint8(rgb))
    img_np = np.array(img)
    concat = Image.new('RGB', (img_np.shape[1] + prediction.size[0], img_np.shape[0]))
    concat.paste(rgb_pil, (0, 0))
    concat.paste(prediction, (img_np.shape[1], 0))
    concat.save(save_path)
    return concat

if __name__ == "__main__":
    cfg = init()

    print("************************ Start Logging ************************")
    print(cfg)
    print("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    print("creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    print_gpu_memory(" ========= After model.cuda() ============")
    if cfg.checkpoint_path == 'none':
        checkpoint = os.path.join(cfg.log_dir, 'checkpoint_iter' + str(cfg.test_iter).zfill(6) + '.pth')
    else:
        checkpoint = cfg.checkpoint_path
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)


    if cfg.dataset == 'all':
        datasets = ['ycbv', 'tudl',  'lmo', 'icbin', 'tless', 'itodd' , 'hb']
        for dataset_name in datasets:
            print('begining evaluation on {} ...'.format(dataset_name))

            save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path,'result_' + dataset_name +'.csv')
            test(model, cfg, save_path, dataset_name, detetion_paths[dataset_name])

            print('saving to {} ...'.format(save_path))
            print('finishing evaluation on {} ...'.format(dataset_name))

    else:
        dataset_name = cfg.dataset
        print('begining evaluation on {} ...'.format(dataset_name))

        save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path,'result_' + dataset_name + f'_scene_01-10_v_DBSCAN_ICP_P2P_corrected'+'.csv')
        test(model, cfg,  save_path, dataset_name, detetion_paths[dataset_name])

        print('saving to {} ...'.format(save_path))
        print('finishing evaluation on {} ...'.format(dataset_name))