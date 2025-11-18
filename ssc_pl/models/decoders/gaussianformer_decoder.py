from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config
from mmseg.registry import MODELS
from gaussianformer.model.head import GaussianRenderHead
# from GaussianOcc.utils import geom
# from GaussianOcc.utils import vox
# from GaussianOcc.utils import basic
# from GaussianOcc.utils import render
import numpy as np
import torch
from ..utils import rasterize_gaussians, prepare_gs_attribute, setup_opengl_proj
import time
from gsplat import rasterization

SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def normalize_quaternion(q):
    # 计算四元数的范数
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    # 归一化四元数
    q_normalized = q / norm
    return q_normalized

class GaussianFormerDecoder(nn.Module):

    def __init__(self,
                 config_path,
                 custom_imports,
                 checkpoint_path=None,
                 embed_dims=128,
                 scales=(4, 8, 16),
                 num_queries=100,
                 freeze=False,
                 use_encoder_deform_att=True):
        super().__init__()
        import_module(custom_imports)
        config = Config.fromfile(config_path)

        self.model = MODELS.build(config.model)
        # self.render_head = GaussianRenderHead(embed_dims, segment_head=True)

        self.Z_final = 50
        self.Y_final = 100
        self.X_final = 100
        scene_centroid_x = 0.0
        scene_centroid_y = 0.0
        scene_centroid_z = 0.0

        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])

        self.register_buffer('scene_centroid', torch.from_numpy(scene_centroid).float())
        self.bounds = (0, 4, 0, 4, 0, 2)
        self.position = 'embedding'
        length_pose_encoding = 3
        self.opt = None

    def forward(self, metas=None, points=None, ms_img_feats=None, voxel_feat=None, pca_matrix=None):
        # TODO: The following is only devised for the GauusianFormer implementation.

        results = {
            'metas': metas,
            'points': points,
            'ms_img_feats': ms_img_feats
        }

        # voxel_feat (B, 100, 100, 50, C)
        # center = torch.tensor([[1.2475, 0.0673, 1.5356]])
        # # vox_grid, Z, Y, X = self.gs_vox_util.get_voxel_grid(cam_center=center, )
        # out_channel = voxel_feat.shape[1]
        # gs_attribute = voxel_feat.permute(0, 4, 3, 2, 1).reshape(-1, out_channel)

        # gs_attribute = voxel_feat.permute(0, 2, 3, 4, 1).reshape(-1, out_channel)
        # geo_feats = torch.sigmoid(gs_attribute[:, -1:])

        # self.model.lifter.update_semantic(gs_attribute)
        # self.model.lifter.update_opacity(geo_feats)
        # self.model.lifter.update_xyz(vox_grid.squeeze(0).to(geo_feats.device))
        # import pdb;
        # pdb.set_trace()




        # UPDATE
        # xyz = self.model.lifter.xyz
        # xyz = safe_sigmoid(xyz)
        # multipliers = torch.tensor([100.0, 100.0, 50.0])
        # coords = xyz * multipliers
        # coords = torch.round(coords)
        # # 25600,3

        # coords = np.clip(coords, [0, 0, 0], [99, 99, 49]).long()
        # voxel_feat = voxel_feat.squeeze(0).permute(1, 2, 3, 0)
        # # import pdb;
        # # pdb.set_trace()
        # extracted_features = voxel_feat[coords[:, 0], coords[:, 1], coords[:, 2], :] # (25600, 12)
        # self.model.lifter.update_semantic(extracted_features)




        # opacity = safe_inverse_sigmoid(torch.sigmoid(self.model.lifter.opacity_head(extracted_features)))
        # scale = safe_inverse_sigmoid(F.softplus(self.model.lifter.scale_head(extracted_features)))
        # rot = normalize_quaternion(self.model.lifter.rot_head(extracted_features))
        # rot[:, 0] = 1

        # self.model.lifter.update_opacity(opacity)
        # self.model.lifter.update_scale(scale)
        # self.model.lifter.update_rot(rot)

        outs = self.model.lifter(**results)

        # print('model.lifter.rep_features', torch.isnan(outs['rep_features']).any())  # 检查 model.lifter.rep_features 是否有 NaN
        # print('model.lifter.rep_features', torch.isinf(outs['rep_features']).any())  # 检查 model.lifter.rep_features 是否有 Inf
        # print('model.lifter.representation', torch.isnan(outs['representation']).any())   # 检查 model.lifter.representation 是否有 NaN
        # print('model.lifter.representation', torch.isinf(outs['representation']).any())   # 检查 model.lifter.representation 是否有 Inf
        # print('GaussianFormer.outs[rep_features].shape: {}'.format(outs['rep_features'].shape))
        # print('GaussianFormer.outs[representation].shape: {}'.format(outs['representation'].shape))

        results.update(outs)
        outs = self.model.encoder(**results)

        # print(f'outs.keys: {outs.keys()}')
        # for i in range(len(outs['representation'])):
        #     print('outs[representation][{}].keys: {}'.format(i, outs['representation'][i].keys()))
        #     print('model.encoder.means', torch.isnan(outs['representation'][i]['gaussian'].means).any())  # 检查 model.encoder.means 是否有 NaN
        #     print('model.encoder.means', torch.isinf(outs['representation'][i]['gaussian'].means).any())   # 检查 model.encoder.means 是否有 Inf
        #     print('outs[representation][{}][gaussian].means.min: {}'.format(i, outs['representation'][i]['gaussian'].means.min()))
        #     print('outs[representation][{}][gaussian].means.max: {}'.format(i, outs['representation'][i]['gaussian'].means.max()))
        #     print('outs[representation][{}][gaussian].means.shape: {}'.format(i, outs['representation'][i]['gaussian'].means.shape))
        #     print('model.encoder.scales', torch.isnan(outs['representation'][i]['gaussian'].scales).any())  # 检查 model.encoder.scales 是否有 NaN
        #     print('model.encoder.scales', torch.isinf(outs['representation'][i]['gaussian'].scales).any())   # 检查 model.encoder.scales 是否有 Inf
        #     print('outs[representation][{}][gaussian].scales.min: {}'.format(i, outs['representation'][i]['gaussian'].scales.min()))
        #     print('outs[representation][{}][gaussian].scales.max: {}'.format(i, outs['representation'][i]['gaussian'].scales.max()))
        #     print('outs[representation][{}][gaussian].scales.shape: {}'.format(i, outs['representation'][i]['gaussian'].scales.shape))
        #     print('model.encoder.rotations', torch.isnan(outs['representation'][i]['gaussian'].rotations).any())  # 检查 model.encoder.rotations 是否有 NaN
        #     print('model.encoder.rotations', torch.isinf(outs['representation'][i]['gaussian'].rotations).any())   # 检查 model.encoder.rotations 是否有 Inf
        #     print('outs[representation][{}][gaussian].rotations.min: {}'.format(i, outs['representation'][i]['gaussian'].rotations.min()))
        #     print('outs[representation][{}][gaussian].rotations.max: {}'.format(i, outs['representation'][i]['gaussian'].rotations.max()))
        #     print('outs[representation][{}][gaussian].rotations.shape: {}'.format(i, outs['representation'][i]['gaussian'].rotations.shape))
        #     print('model.encoder.opacities', torch.isnan(outs['representation'][i]['gaussian'].opacities).any())  # 检查 model.encoder.opacities 是否有 NaN
        #     print('model.encoder.opacities', torch.isinf(outs['representation'][i]['gaussian'].opacities).any())   # 检查 model.encoder.opacities 是否有 Inf
        #     print('outs[representation][{}][gaussian].opacities.min: {}'.format(i, outs['representation'][i]['gaussian'].opacities.min()))
        #     print('outs[representation][{}][gaussian].opacities.max: {}'.format(i, outs['representation'][i]['gaussian'].opacities.max()))
        #     print('outs[representation][{}][gaussian].opacities.shape: {}'.format(i, outs['representation'][i]['gaussian'].opacities.shape))
        #     print('model.encoder.semantics', torch.isnan(outs['representation'][i]['gaussian'].semantics).any())  # 检查 model.encoder.semantics 是否有 NaN
        #     print('model.encoder.semantics', torch.isinf(outs['representation'][i]['gaussian'].semantics).any())   # 检查 model.encoder.semantics 是否有 Inf
        #     print('outs[representation][{}][gaussian].semantics.min: {}'.format(i, outs['representation'][i]['gaussian'].semantics.min()))
        #     print('outs[representation][{}][gaussian].semantics.max: {}'.format(i, outs['representation'][i]['gaussian'].semantics.max()))
        #     print('outs[representation][{}][gaussian].semantics.shape: {}'.format(i, outs['representation'][i]['gaussian'].semantics.shape))

        results.update(outs)

        # outs = self.render_head(**results)
        # results.update(outs)

        outs = self.model.head(**results)
        # print(f'outs.keys: {outs.keys()}')
        # for i in range(len(outs['pred_occ'])):
        #     print('model.head.pred_occ', torch.isnan(outs['pred_occ'][i]).any())   # 检查 model.head.pred_occ 是否有 NaN
        #     print('model.head.pred_occ', torch.isinf(outs['pred_occ'][i]).any())   # 检查 model.head.pred_occ 是否有 Inf
        #     print('outs[pred_occ][{}].shape: {}'.format(i, outs['pred_occ'][i].shape))
        # print('model.head.sampled_xyz', torch.isnan(outs['sampled_xyz']).any())   # 检查 model.head.sampled_xyz 是否有 NaN
        # print('model.head.sampled_xyz', torch.isinf(outs['sampled_xyz']).any())   # 检查 model.head.sampled_xyz 是否有 Inf
        # print('outs[sampled_xyz].shape: {}'.format(outs['sampled_xyz'].shape))
        # print('model.head.occ_mask', torch.isnan(outs['occ_mask']).any())   # 检查 model.head.occ_mask 是否有 NaN
        # print('model.head.occ_mask', torch.isinf(outs['occ_mask']).any())   # 检查 model.head.occ_mask 是否有 Inf
        # print('outs[occ_mask].shape: {}'.format(outs['occ_mask'].shape))
        results.update(outs)
        return results