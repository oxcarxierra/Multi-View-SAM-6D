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
        coarse_endpoints = []
        pred_Rs = []
        pred_Ts = []
        pred_scores = []
        for i, data in enumerate(dataloder):
            print("=====================> Processing {}th data".format(i))
            # data = dict_keys(['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'scene_id', 'img_id', 'seg_time']) -> for same scene & image show all proposals
            torch.cuda.synchronize()
            end = time.time()

            for key in data:
                data[key] = data[key].cuda()
            n_instance = data['pts'].size(1)
            n_batch = int(np.ceil(n_instance/bs))
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch-1 else (j+1) * bs
                obj = data['obj'][0][start_idx:end_idx].reshape(-1)

                # process inputs
                inputs = {}
                inputs['pts'] = data['pts'][0][start_idx:end_idx].contiguous()
                inputs['rgb'] = data['rgb'][0][start_idx:end_idx].contiguous()
                inputs['rgb_choose'] = data['rgb_choose'][0][start_idx:end_idx].contiguous()
                inputs['model'] = data['model'][0][start_idx:end_idx].contiguous()
                inputs['dense_po'] = dense_po[obj].contiguous()
                inputs['dense_fo'] = dense_fo[obj].contiguous()

                # make predictions
                with torch.no_grad():
                    end_points, _, _, _, _, _ = model.singleview_coarse_point_matching(inputs)
                    # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo', 'init_R', 'init_t'])

                coarse_endpoints.append(end_points)
                # merge pointclouds by batch

                if len(coarse_endpoints) != cfg.multi_view_count:
                    continue    

                print(f"=====================> Multiview estimation with {len(coarse_endpoints)} images")
                from collections import defaultdict

                count_by_obj = defaultdict(int)
                for singleview_data in coarse_endpoints:
                    import pdb; pdb.set_trace()
                    obj_ids = set(singleview_data['data']['obj_id'][0].tolist())
                    for obj_id in obj_ids:
                        count_by_obj[obj_id] += 1

                valid_obj_ids = [obj_id for obj_id, count in count_by_obj.items() if count == len(coarse_endpoints)]

                print(f"Found {len(valid_obj_ids)} objects - {valid_obj_ids}")

                for obj_id in valid_obj_ids:
                    merged_pts_list = []
                    merged_rgb_list = []
                    merged_rgb_choose_list = []
                    T_w2c_list = []
                    init_R_list = []
                    init_t_list = []

                    for singleview_data in coarse_endpoints:
                        data = singleview_data['data']
                        obj_ids = data['obj_id'][0].tolist()

                        indices = [i for i, oid in enumerate(obj_ids) if oid == obj_id]
                        if not indices:
                            continue
                        scores = data['score'][0][indices]
                        best_idx = indices[torch.argmax(scores).item()]

                        pts = data['pts'][0][best_idx]
                        rgb = data['rgb'][0][best_idx]
                        rgb_choose = data['rgb_choose'][0][best_idx]
                        cam_R_w2c = data['cam_R_w2c'][0]
                        cam_t_w2c = data['cam_t_w2c'][0]

                        T_w2c = torch.eye(4).cuda()
                        T_w2c[:3, :3] = cam_R_w2c
                        T_w2c[:3, 3] = cam_t_w2c
                        T_c2w = torch.inverse(T_w2c)

                        R_c2w = T_c2w[:3, :3]
                        t_c2w = T_c2w[:3, 3]

                        pts_world = (R_c2w @ pts.T + t_c2w.unsqueeze(-1)).T

                        merged_pts_list.append(pts_world)
                        merged_rgb_list.append(rgb)
                        merged_rgb_choose_list.append(rgb_choose)
                        T_w2c_list.append(T_w2c)

                    if not merged_pts_list:
                        continue

                    merged_pts = torch.cat(merged_pts_list, dim=0).unsqueeze(0)
                    merged_rgb = torch.cat(merged_rgb_list, dim=0).unsqueeze(0)
                    merged_rgb_choose = torch.cat(merged_rgb_choose_list, dim=0).unsqueeze(0)

                    # Random sampling
                    total_points = merged_pts.shape[1]
                    sampled_idx = torch.randperm(total_points)[:total_points // cfg.multi_view_count]
                    merged_pts = merged_pts[:, sampled_idx]
                    merged_rgb = merged_rgb[:, sampled_idx]
                    merged_rgb_choose = merged_rgb_choose[:, sampled_idx]

                    # Prepare inputs
                    inputs = {
                        'pts': merged_pts,
                        'rgb': merged_rgb,
                        'rgb_choose': merged_rgb_choose,
                        'model': torch.tensor([obj_id], dtype=torch.long).cuda(),
                        'dense_po': coarse_endpoints[0]['dense_po'][0].unsqueeze(0),
                        'dense_fo': coarse_endpoints[0]['dense_fo'][0].unsqueeze(0),
                    }

                    with torch.no_grad():
                        end_points = model.multiview_fine_point_matching(inputs)

                    R_final = end_points['final_R'][0]
                    t_final = end_points['final_t'][0]

                    # Inverse transform to each camera's frame
                    for i, T_w2c in enumerate(T_w2c_list):
                        cam_R_w2c = T_w2c[:3, :3]
                        cam_t_w2c = T_w2c[:3, 3]

                        R_img = cam_R_w2c @ R_final
                        t_img = cam_R_w2c @ t_final + cam_t_w2c

                        scene_id = coarse_endpoints[i]['data']['scene_id'].item()
                        img_id = coarse_endpoints[i]['data']['img_id'].item()
                        line = ','.join((
                            str(scene_id),
                            str(img_id),
                            str(obj_id),
                            str(1.0),
                            ' '.join((str(v) for v in R_img.flatten().tolist())),
                            ' '.join((str(v * 1000) for v in t_img.tolist())),
                            '0.0\n'
                        ))
                        lines.append(line)

                coarse_endpoints.clear()  # Reset buffer after multiview processing

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





