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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))


detetion_paths = {
    'ycbv': '../Instance_Segmentation_Model/log/sam/result_ycbv.json',
    'tudl': '../Instance_Segmentation_Model/log/sam/result_tudl.json',
    'tless': '../Instance_Segmentation_Model/log/sam/result_tless.json',
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
                        default="config/base.yaml",
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

    cfg.multi_view_count = 5 # number of views to be used as multiview input

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    return cfg

def test(model, cfg, save_path, dataset_name, detetion_path):
    model.eval()
    bs = cfg.test_dataloader.bs

    # build dataloader
    dataset = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset.BOPTestset(cfg.test_dataset, dataset_name, detetion_path)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg.test_dataloader.num_workers,
            shuffle=cfg.test_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.test_dataloader.drop_last,
            pin_memory=cfg.test_dataloader.pin_memory
        )

    # prepare for target objects
    all_tem, all_tem_pts, all_tem_choose = dataset.get_templates()
    with torch.no_grad():
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    lines = []
    with tqdm(total=len(dataloder)) as t:
        multiview_datas = []
        for data_idx, data in enumerate(dataloder):
            print("=====================> Processing {}th data".format(data_idx))
            # data = dict_keys(['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'scene_id', 'img_id', 'seg_time']) -> for same scene & image show all proposals
            torch.cuda.synchronize()
            end = time.time()

            for key in data:
                data[key] = data[key].cuda()

            # Commented batch processing
            # n_instance = data['pts'].size(1)
            # n_batch = int(np.ceil(n_instance/bs))
            # for j in range(n_batch): 
            # start_idx = j * bs
            # end_idx = n_instance if j == n_batch-1 else (j+1) * bs
            # obj = data['obj'][0][start_idx:end_idx].reshape(-1)

            # start_idx = 0
            # end_idx = data['pts'].size(1)
            # obj = data['obj'][0][start_idx:end_idx].reshape(-1)
            # import pdb; pdb.set_trace()
            # # process inputs
            # inputs = {}
            # inputs['pts'] = data['pts'][0][start_idx:end_idx].contiguous()
            # inputs['rgb'] = data['rgb'][0][start_idx:end_idx].contiguous()
            # inputs['rgb_choose'] = data['rgb_choose'][0][start_idx:end_idx].contiguous()
            # inputs['model'] = data['model'][0][start_idx:end_idx].contiguous()
            # inputs['dense_po'] = dense_po[obj].contiguous()
            # inputs['dense_fo'] = dense_fo[obj].contiguous()

            # make predictions
            # with torch.no_grad():
            #     end_points, fps_idx_m, fps_idx_o = model.singleview_coarse_point_matching(inputs)
            # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo', 'init_R', 'init_t'])

            single_image_proposals = []
            for p in range(len(data['pts'][0])):
                proposal = {}
                for key in data:
                    if key in ['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'cam_R_w2c', 'cam_t_w2c']:
                        proposal[key] = data[key][0][p]
                    else:
                        proposal[key] = data[key][0]  # just take [0] for image-level data
                # for key in end_points:
                #     proposal[key] = end_points[key][p]
                single_image_proposals.append(proposal)

            multiview_datas.append(single_image_proposals)
            # multiview_datas = [[proposal1, proposal2, ...], [proposal1, proposal2, ...], ...]
            # proposal is a object with key ['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'cam_K', 'cam_R_w2c', 'cam_t_w2c', 'scene_id', 'img_id', 'seg_time', 'dense_po', 'dense_fo', 'init_R', 'init_t']

            if len(multiview_datas) != cfg.multi_view_count:
                continue    

            print(f"=====================> Multiview estimation with {len(multiview_datas)} images")
            all_proposals = sum(multiview_datas, [])  # flatten
            
            from sklearn.cluster import AffinityPropagation
            from collections import defaultdict

            # Compute similarity matrix using negative 3D center distances
            centers = []
            for proposal in all_proposals:
                center_cam = proposal['pts'].mean(dim=0)  # (3,)
                R_w2c = proposal['cam_R_w2c'].squeeze(0)  # (3, 3)
                t_w2c = proposal['cam_t_w2c'].squeeze(0) / 1000.0  # (3,)
                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c.view(3, 1)
                center_world = (R_c2w @ center_cam.view(3, 1) + t_c2w).view(-1).cpu().numpy()
                centers.append(center_world)
            centers = np.stack(centers)
            N = len(centers)
            S = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    sim = -dist  # Negative distance for AffinityPropagation
                    S[i, j] = S[j, i] = sim
            np.fill_diagonal(S, np.median(S))  # Set diagonal preference

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

            cluster_results = {}
            ## Step 2 : Inference multiview pose estimation for each objects with best ID
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
                total_points = all_pts.shape[0]
                sampled_idx = torch.randperm(total_points)[:2048]
                merged_pts = all_pts[sampled_idx]  # shape: (2048, 3)

                
                batch_inputs = {
                    'pts': [],
                    'rgb': [],
                    'rgb_choose': [],
                    'model': [],
                    'dense_po': [],
                    'dense_fo': []
                }

                for proposal in proposals_for_obj:
                    obj = proposal['obj'][0].item()
                    
                    batch_inputs['pts'].append(merged_pts.unsqueeze(0))
                    batch_inputs['rgb'].append(proposal['rgb'].unsqueeze(0))
                    batch_inputs['rgb_choose'].append(proposal['rgb_choose'].unsqueeze(0))
                    batch_inputs['model'].append(proposal['model'].unsqueeze(0))
                    batch_inputs['dense_po'].append(dense_po[obj].unsqueeze(0))
                    batch_inputs['dense_fo'].append(dense_fo[obj].unsqueeze(0))

                for key in batch_inputs:
                    batch_inputs[key] = torch.cat(batch_inputs[key], dim=0)

                with torch.no_grad():
                    end_points = model(batch_inputs)

                pred_scores = end_points['pred_pose_score']
                best_idx = torch.argmax(pred_scores).item()
                best_proposal = proposals_for_obj[best_idx]
                obj_id = best_proposal['obj_id'].item()
                R_final = end_points['pred_R'][best_idx]
                t_final = end_points['pred_t'][best_idx]
                cluster_results[label] = {
                    'obj_id': obj_id,
                    'score': pred_scores[best_idx].item(),
                    'R': R_final,
                    't': t_final
                }

                for i in range(len(proposals_for_obj)):
                    print(f"→ [Cluster {label}] Proposal {i}: obj_id: {proposals_for_obj[i]['obj_id'].item()}, score: {proposals_for_obj[i]['score'].item():.4f}")
                print(f"→ [Cluster {label}] Best Proposal: obj_id: {obj_id}, score: {pred_scores[best_idx].item():.4f}")

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

                    R_img = R_w2c @ R_final
                    t_img = (R_w2c @ t_final.view(3, 1) + t_w2c.view(3, 1)).view(-1)

                    image_time = 0
                    line = ','.join((
                        str(scene_id),
                        str(img_id),
                        str(obj_id),
                        f'{pred_scores[best_idx].item():.6f}',
                        ' '.join(f'{v:.8f}' for v in R_img.flatten().tolist()),
                        ' '.join(f'{v * 1000:.8f}' for v in t_img.tolist()),
                        str(image_time)
                    )) + '\n'
                    lines.append(line)

                    with open(save_path, 'w+') as f:
                        f.writelines(lines)

            # Clear data structures
            cluster_proposals.clear()
            multiview_datas.clear()

    with open(save_path, 'w+') as f:
        f.writelines(lines)

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





