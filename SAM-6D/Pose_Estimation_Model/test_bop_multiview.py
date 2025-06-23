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
# from draw_utils import draw_detections

MAX_PROPOSALS = 10
BATCH_LIMIT = 4  # adjust if needed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))


detetion_paths = {
    'ycbv': '../Instance_Segmentation_Model/log/sam/result_ycbv.json',
    'tudl': '../Instance_Segmentation_Model/log/sam/result_tudl.json',
    'tless': '../Instance_Segmentation_Model/log/sam/result_tless_ism.json',
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
                        default=False,
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

def test(model, cfg, save_path, dataset_name, detetion_path):
    model.eval()
    bs = cfg.test_dataloader.bs
    cluster_summary_lines = []

    def identity_collate(batch):
        return batch[0]

    dataset = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset.BOPMultiviewTestset(cfg.test_dataset, dataset_name, detetion_path)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
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
        for group_idx, data in enumerate(dataloder):
            print(f"=====================> Processing {group_idx+1}th multiview batch")

            torch.cuda.synchronize()
            end = time.time()

            # ----------------------------------------------------------
            # Step 0: Read results from ISM and save to multiview_datas
            # ----------------------------------------------------------
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
            
            # ----------------------------------------------------------
            # Step 1: Proposal Matching
            # ----------------------------------------------------------

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

            # ----------------------------------------------------------
            # Step 2 : Inference multiview pose estimation for each cluster
            # ----------------------------------------------------------

            cluster_results = {}
            obj_ids = []
            for label, proposals_for_obj in cluster_proposals.items():
                merged_pts_list = []
                input_list = []
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

                all_pts = torch.cat(merged_pts_list, dim=0)
                filtered_pts = remove_outliers_lof(all_pts, n_neighbors=20, contamination=0.01)
                n_points = filtered_pts.shape[0]
                sampled_idx = torch.randperm(n_points)[:proposals_for_obj[0]['pts'].shape[0]]
                merged_pts = filtered_pts[sampled_idx]  # shape: (2048, 3)

                cluster_obj_id = max(proposals_for_obj, key=lambda p: p['score'].item())['obj_id'].item()
                model_for_cluster_obj_id = next(proposal['model'] for proposal in proposals_for_obj if proposal['obj_id'].item() == cluster_obj_id)
                obj_ids.append(cluster_obj_id)

                # === PEM Inference with Memory-Safe Batching ===
                cluster_outputs = {
                    'pred_pose_score': [],
                    'pred_R': [],
                    'pred_t': [],
                    'score': []
                }

                used_proposals = []
                N = len(proposals_for_obj)

                for i in range(0, N, BATCH_LIMIT):

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

                    for key in batch_inputs:
                        batch_inputs[key] = torch.cat(batch_inputs[key], dim=0)

                    with torch.no_grad():
                        sub_output = model(batch_inputs)

                    for k in cluster_outputs:
                        cluster_outputs[k].append(sub_output[k].cpu())  # keep on CPU

                    del batch_inputs, sub_output
                    torch.cuda.empty_cache()
                
                # ----------------------------------------------------------
                # Step 3 : Object Classification
                # ----------------------------------------------------------
                end_points = {k: torch.cat(v, dim=0) for k, v in cluster_outputs.items()}
                
                # Debug check
                assert len(used_proposals) == end_points['pred_pose_score'].shape[0], \
                    f"Mismatched proposal/output sizes: {len(used_proposals)} vs {end_points['pred_pose_score'].shape[0]}"

                # Convert only once, store result
                pred_scores = end_points['pred_pose_score'].cuda() * end_points['score'].cuda()

                best_idx = torch.argmax(pred_scores).item()
                best_proposal = proposals_for_obj[best_idx]
                best_proposal_obj_id = best_proposal['obj_id'].item()

                R_c2w = best_proposal['cam_R_w2c'].squeeze(0).T
                t_c2w = -R_c2w @ (best_proposal['cam_t_w2c'].squeeze(0) / 1000.0).view(3, 1)
                R_final = R_c2w @ end_points['pred_R'].cuda()[best_idx]
                t_final = (R_c2w @ end_points['pred_t'].cuda()[best_idx].view(3, 1) + t_c2w).view(-1)
                
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
                
            # ----------------------------------------------------------
            # Step 4:  Write final pose of each cluster to every image in multiview_datas (sorted by img_id)
            # ----------------------------------------------------------

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
                    
                    if cfg.visualization:
                        print(f"Visualizing scene {scene_id}, image {img_id}, cluster {label}, object {obj_id}")
                        vis_dir = save_path.replace('result_tless.csv',f"pem_visualization/scene_{scene_id:06d}/")
                        os.makedirs(os.path.dirname(vis_dir), exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"group{group_idx+1}_img{img_id}_cluster{label}_obj{obj_id}.png")
                        img_path = BASE_DIR + "/../Data/BOP/tless/test_primesense/" + str(scene_id).zfill(6) + "/rgb/" + str(img_id).zfill(6) + ".png"
                        cad_path = BASE_DIR + "/../Data/BOP/tless/models_cad/obj_" + str(obj_id).zfill(6) + ".ply"
                        img = np.array(Image.open(img_path))
                        import trimesh
                        mesh = trimesh.load_mesh(cad_path)
                        model_points = mesh.sample(512).astype(np.float32)
                        K_ = proposal['cam_K'].cpu().numpy()
                        vis_img = visualize(img, [merged_pts_camera_frame.cpu().numpy()], [R_img.cpu().numpy()], [t_img.cpu().numpy()], model_points, K_, vis_path)
            t.update(1)
                    
            cluster_proposals.clear()
            multiview_datas.clear()

    with open(save_path, 'w+') as f:
        f.writelines(lines)

def visualize(rgb, pointcloud, pred_rot, pred_trans, model_points, K, save_path):
    from draw_utils import draw_detections
    img = draw_detections(rgb, pointcloud, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    print(f"Visualization saved to {save_path}")
    print(img)
    return img

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
        save_path = os.path.join(save_path,'result_' + dataset_name +'.csv')
        test(model, cfg,  save_path, dataset_name, detetion_paths[dataset_name])

        print('saving to {} ...'.format(save_path))
        print('finishing evaluation on {} ...'.format(dataset_name))