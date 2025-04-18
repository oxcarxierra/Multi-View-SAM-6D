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
        multiview_data = []
        pred_Rs = []
        pred_Ts = []
        pred_scores = []
        for i, data in enumerate(dataloder):
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
                    end_points = model.coarse_point_matching_forward(inputs)
                    # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo', 'init_R', 'init_t'])

                coarse_result = {
                    'data': data,
                    'init_R': end_points['init_R'], # [n_instance, 3, 3]
                    'init_t': end_points['init_t'], # [n_instance, 3]
                }

                multiview_data.append(coarse_result)
                # merge pointclouds by batch

                if len(multiview_data) == cfg.multi_view_count:
                    import pdb; pdb.set_trace()
                    # merge pointclouds
                    merged_results, _, _, _ = merge_pointclouds(multiview_data)

                    for merged_pc, obj_id, K_list, T_w2c_list in merged_results:
                        # TODO we need to fix this
                        inputs = {
                            'pts': merged_pc,
                            'model': torch.tensor([obj_id], dtype=torch.long).cuda(),
                            'dense_po': dense_po[obj_id].unsqueeze(0),
                            'dense_fo': dense_fo[obj_id].unsqueeze(0),
                        }

                        with torch.no_grad():
                            end_points = model.multiview_finepoint_matching_forward(inputs)

                        R_final = end_points['final_R'][0]  # (3, 3)
                        t_final = end_points['final_t'][0]  # (3,)

                        # Inverse transform to each camera's frame
                        for i, entry in enumerate(multiview_data):
                            data = entry['data']
                            T_w2c = T_w2c_list[i]  # (4, 4)
                            cam_R_w2c = T_w2c[:3, :3]  # (3, 3)
                            cam_t_w2c = T_w2c[:3, 3]  # (3,)

                            # TODO fix here
                            # world to camera: X_cam = R_cam * (R_final * X + t_final) + t_cam
                            # R_img = R_cam @ R_final
                            # t_img = R_cam @ t_final + t_cam

                            scene_id = data['scene_id'].item()
                            img_id = data['img_id'].item()
                            line = ','.join((
                                str(scene_id),
                                str(img_id),
                                str(obj_id),
                                str(1.0),  # dummy score
                                ' '.join((str(v) for v in R_img.flatten().tolist())),
                                ' '.join((str(v * 1000) for v in t_img.tolist())),
                                '0.0\n'
                            ))
                            lines.append(line)

            t.set_description(
                "Test [{}/{}]".format(i+1, len(dataloder))
            )
            t.update(1)

    with open(save_path, 'w+') as f:
        f.writelines(lines)

def merge_pointclouds(multiview_data): 
    # How to select same object from different views? 
    # Assume all objects are different, so we can just use the object id to match them.
    from collections import defaultdict

    merged_results = []
    count_by_obj = defaultdict(int)
    num_points = multiview_data[0]['data']['pts'][0].shape[0]  # Number of points in the first view

    # 1. Count unique obj_ids per view to avoid duplicate counting
    for singleview_data in multiview_data:
        obj_ids = set(singleview_data['data']['obj_id'][0].tolist())
        for obj_id in obj_ids:
            count_by_obj[obj_id] += 1

    # 2. Filter obj_ids that appear in all views
    valid_obj_ids = [obj_id for obj_id, count in count_by_obj.items() if count == len(multiview_data)]

    # 3. Iterate through each valid obj_id and merge point clouds
    for obj_id in valid_obj_ids:
        pts_list = []
        K_list = []
        T_w2c_list = []

        for singleview_data in multiview_data:
            data = singleview_data['data']
            obj_ids = data['obj_id'][0].tolist()

            if obj_id not in obj_ids:
                continue

            # Find all indices where obj_id matches
            indices = [i for i, oid in enumerate(obj_ids) if oid == obj_id]
            scores = data['score'][0][indices]
            best_idx = indices[torch.argmax(scores).item()]

            R_init = singleview_data['init_R'][0][best_idx]
            t_init = singleview_data['init_t'][0][best_idx]
            pts = data['pts'][0][best_idx]
            pts = pts.unsqueeze(0) if pts.ndim == 1 else pts  # ensure shape (N, 3)

            K = data['cam_K'][0]
            cam_R_w2c = data['cam_R_w2c'][0]
            cam_t_w2c = data['cam_t_w2c'][0]

            # World-frame 변환
            T_w2c = torch.cat((cam_R_w2c, cam_t_w2c), dim=1)  # (3, 4)
            T_w2c = torch.cat((T_w2c, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()), dim=0)
            T_c2w = torch.inverse(T_w2c)
            R_c2w = T_c2w[:3, :3]
            t_c2w = T_c2w[:3, 3]

            # init R, t 적용 → camera frame → world frame
            # transformed_pts = (R_init @ pts.T + t_init.unsqueeze(-1)).T  # (N, 3)
            pts_base_frame = (R_c2w @ pts.T + t_c2w.unsqueeze(-1)).T  # (N, 3)

            pts_list.append(pts_base_frame)
            K_list.append(K)
            T_w2c_list.append(T_w2c)

        if pts_list:
            # Merge all points first
            merged_pc = torch.cat(pts_list, dim=0)
            indices = torch.randperm(merged_pc.shape[0])[:num_points]
            merged_pc = merged_pc[indices]
            merged_pc = merged_pc.unsqueeze(0)  # (1, num_points, 3)
            merged_results.append((merged_pc, obj_id, K_list, T_w2c_list))

    return merged_results, pc_list, K_list, T_w2c_list

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





