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
        for i, data in enumerate(dataloder):
            print("=====================> Processing {}th data".format(i))
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

            start_idx = 0
            end_idx = data['pts'].size(1)
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
                end_points, fps_idx_m, fps_idx_o = model.singleview_coarse_point_matching(inputs)
                # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo', 'init_R', 'init_t'])

            single_image_proposals = []
            for p in range(len(data['pts'][0])):
                proposal = {}
                for key in data:
                    if key in ['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'cam_R_w2c', 'cam_t_w2c']:
                        proposal[key] = data[key][0][p]
                    else:
                        proposal[key] = data[key][0]  # just take [0] for image-level data
                for key in end_points:
                    proposal[key] = end_points[key][p]
                proposal['fps_idx_m'] = fps_idx_m[p]
                proposal['fps_idx_o'] = fps_idx_o[p]
                single_image_proposals.append(proposal)

            multiview_datas.append(single_image_proposals)
            # multiview_datas = [[proposal1, proposal2, ...], [proposal1, proposal2, ...], ...]
            # proposal is a object with key ['pts', 'rgb', 'rgb_choose', 'obj', 'model', 'obj_id', 'score', 'cam_K', 'cam_R_w2c', 'cam_t_w2c', 'scene_id', 'img_id', 'seg_time', 'dense_po', 'dense_fo', 'init_R', 'init_t']

            if len(multiview_datas) != cfg.multi_view_count:
                continue    

            print(f"=====================> Multiview estimation with {len(multiview_datas)} images")
            all_proposals = sum(multiview_datas, [])  # flatten
            ## Step 1 : Match the objects in multiview images

            # Sum scores for each obj_id
            obj_score_sum = {}
            for proposal in all_proposals:
                obj_id = proposal['obj_id'].item()
                score = proposal['score'].item()
                obj_score_sum[obj_id] = obj_score_sum.get(obj_id, 0) + score

            # top 4 object ids with highest scores - need to be changed
            top_obj_ids = sorted(obj_score_sum.items(), key= lambda x: x[1], reverse=True)[:4] 
            top_obj_ids = [obj_id for obj_id, _ in top_obj_ids]

            print(f"Found {len(top_obj_ids)} objects - {top_obj_ids}")
            print(obj_score_sum)
            
            for obj_id in top_obj_ids:
                proposals_for_obj = [p for p in all_proposals if p['obj_id'].item() == obj_id]
                if not proposals_for_obj:
                    continue
                best_proposal = max(proposals_for_obj, key=lambda p: p['score'].item())

                merged_pts_list = []
                T_w2c_list = []

                for view_proposals in multiview_datas:
                    proposals_in_view = [p for p in view_proposals if p['obj_id'].item() == obj_id]
                    if not proposals_in_view:
                        continue
                    best_in_view = max(proposals_in_view, key=lambda p: p['score'].item())
                    
                    pts = best_in_view['pts']
                    R_w2c = best_in_view['cam_R_w2c'].squeeze(0)
                    t_w2c = best_in_view['cam_t_w2c'].squeeze(0) / 1000.0
                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c.view(3, 1)

                    pts_world = (R_c2w @ pts.T + t_c2w).T

                    # Export merged_pts as .npy file
                    obj_save_path = os.path.join(save_path,'individual_pcls',f"img_{best_in_view['img_id'].item()}_obj_{obj_id}.npy")
                    np.save(obj_save_path.replace('/result_tless.csv',''), pts_world.cpu().numpy())

                    merged_pts_list.append(pts_world)
                    T_w2c_list.append(t_w2c)

                if not merged_pts_list:
                    continue

                merged_pts = torch.cat(merged_pts_list, dim=0)  # (N*2048, 3)
                total_points = merged_pts.shape[0]
                sampled_idx = torch.randperm(total_points)[:2048]
                merged_pts = merged_pts[sampled_idx].unsqueeze(0)  # (1, 2048, 3)

                # initial pose proposals to world frame
                import pdb; pdb.set_trace()
                best_proposal_R_w2c = best_proposal['cam_R_w2c'].squeeze(0)
                best_proposal_t_w2c = best_proposal['cam_t_w2c'].squeeze(0) / 1000.0
                best_proposal_R_c2w = best_proposal_R_w2c.T
                best_proposal_t_c2w = -best_proposal_R_w2c @ best_proposal_t_w2c.view(3, 1)
                init_R_world = best_proposal_R_c2w @ best_proposal['init_R']
                init_t_world = (best_proposal_R_c2w @ best_proposal['init_t'].view(3, 1) + best_proposal_t_c2w).view(-1)

                inputs = {
                    'pts': merged_pts,
                    'rgb': best_proposal['rgb'].unsqueeze(0),
                    'rgb_choose': best_proposal['rgb_choose'].unsqueeze(0),
                    'model': best_proposal['model'].unsqueeze(0),
                    'dense_po': best_proposal['dense_po'].unsqueeze(0),
                    'dense_fo': best_proposal['dense_fo'].unsqueeze(0),
                    'init_R': init_R_world.unsqueeze(0),
                    'init_t': init_t_world.unsqueeze(0),
                }

                # Export merged_pts as .npy file
                obj_save_path = os.path.join(save_path, 'merged_pcls', f"merged_{multiview_datas[0][0]['img_id'].item()}-{multiview_datas[-1][0]['img_id'].item()}_obj_{obj_id}.npy")
                np.save(obj_save_path.replace('/result_tless.csv',''), merged_pts[0].cpu().numpy())

                fps_idx_m = best_proposal['fps_idx_m'].unsqueeze(0)
                fps_idx_o = best_proposal['fps_idx_o'].unsqueeze(0)

                with torch.no_grad():
                    end_points = model.multiview_fine_point_matching(inputs, fps_idx_m, fps_idx_o)

                R_final = end_points['pred_R'][0]
                t_final = end_points['pred_t'][0]
                print(f"Prediction score is : {end_points['pred_pose_score'][0]}")

                # Write results : inverse transform to each camera's frame

                # for j, T_w2c in enumerate(T_w2c_list):
                #     cam_R_w2c = T_w2c[:3, :3]
                #     cam_t_w2c = T_w2c[:3, 3]

                #     R_img = cam_R_w2c @ R_final
                #     t_img = cam_R_w2c @ t_final + cam_t_w2c

                #     scene_id = multiview_datas[j][0]['scene_id'].item()
                #     img_id = multiview_datas[j][0]['img_id'].item()
                #     obj_id = multiview_datas[j][0]['obj_id'].item()
                #     line = ','.join((
                #         str(scene_id),
                #         str(img_id),
                #         str(obj_id),
                #         str(1.0),
                #         ' '.join((str(v) for v in R_img.flatten().tolist())),
                #         ' '.join((str(v * 1000) for v in t_img.tolist())),
                #         '0.0\n'
                #     ))
                #     lines.append(line)

            multiview_datas.clear()  # Reset buffer after multiview processing
            import pdb; pdb.set_trace()
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





