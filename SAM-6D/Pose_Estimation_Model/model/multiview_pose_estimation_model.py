import torch
import torch.nn as nn

from feature_extraction import ViTEncoder, Multiview_ViTEncoder
from coarse_point_matching import CoarsePointMatching
from fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = Multiview_ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching)

    def forward(self, end_points):
        # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo'])
        # end_points['pts'].shape = torch.Size([7, 2048, 3])
        # end_points['rgb'].shape = torch.Size([7, 3, 224, 224])
        # end_points['rgb_choose'].shape = torch.Size([7, 2048])
        # end_points['model'].shape = torch.Size([7, 1024, 3])
        # end_points['dense_po'].shape = torch.Size([7, 2048, 3])
        # end_points['dense_fo'].shape = torch.Size([7, 2048, 256])
        dense_pm, dense_fm, dense_po, dense_fo, radius = self.feature_extraction(end_points)

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(
            dense_pm, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(
            dense_po, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # coarse_point_matching
        end_points = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, end_points,
        )
        # end_points = dict_keys(['pts', 'rgb', 'rgb_choose', 'model', 'dense_po', 'dense_fo', 'init_R', 'init_t'])
        # end_points['init_R'].shape = torch.Size([7, 3, 3])
        # end_points['init_t'].shape = torch.Size([7, 3])
        
        # fine_point_matching
        end_points = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )
        # end_points['pred_R'].shape = torch.Size([7, 3, 3])
        # end_points['pred_t'].shape = torch.Size([7, 3])
        # end_points['pred_pose_score'].shape = torch.Size([7])
        
        return end_points