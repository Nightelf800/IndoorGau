import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import DeformableTransformerLayer
from ..utils import (flatten_multi_scale_feats, get_level_start_index, index_fov_back_to_voxels,
                     nlc_to_nchw)


class VoxelProposalLayerMultiBS(nn.Module):
    def __init__(self, embed_dims, scene_shape, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.attn = DeformableTransformerLayer(embed_dims, num_heads, num_levels, num_points)
        self.scene_shape = scene_shape

    # TODO 当前场景仍然可以从历史帧场景中额外获得，虽不出现在当前视角下，但是出现在历史视角下的voxel，从而利用历史帧信息初始化这些voxel特征
    def forward(self, scene_embed, feats, scene_pos=None, vol_pts=None, ref_pix=None, ret_mask=False):
        # scene_embed   # (2,129600,64)
        # feats         # [(2,64,120,160),(2,64,60,80),(2,64,30,40)]
        # scene_pos     # (2,129600,64)
        # vol_pts       # (2,307200,3)                                  # 图像像素投影至3D体素空间的坐标
        # ref_pix       # (2,129600,2)
        B = scene_embed.shape[0]
        keep = ((vol_pts[..., 0] >= 0) & (vol_pts[..., 0] < self.scene_shape[0]) &
                (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] < self.scene_shape[1]) &
                (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] < self.scene_shape[2]))  # keep # (2,307200)
        ## 原本仅支持batch_size=1
        # assert vol_pts.shape[0] == 1
        # geom = vol_pts.squeeze()[keep.squeeze()]
        ## 新版支持batch_size>=2
        outs = []
        masks = []
        for i in range(B):
            # prepare inputs for the current sample i
            keep_i = keep[i:i + 1, ...]
            vol_pts_i = vol_pts[i:i + 1, ...]
            scene_embed_i = scene_embed[i:i + 1, ...]
            feats_i = [x[i:i + 1, ...] for x in feats]
            scene_pos_i = scene_pos[i:i + 1, ...]
            ref_pix_i = ref_pix[i:i + 1, ...]
            geom = vol_pts_i.squeeze()[keep_i.squeeze()]  # 图像中每个像素对应的，在当前场景内的，体素坐标
            pts_mask_i = torch.zeros(self.scene_shape, device=scene_embed.device, dtype=torch.bool)  # (60,60,36)
            pts_mask_i[geom[:, 0], geom[:, 1], geom[:, 2]] = True  # 用于筛选在深度图中出现的体素，VPL只会计算物体表面特征
            masks.append(pts_mask_i)
            pts_mask_i = pts_mask_i.flatten()
            feat_flatten, shapes = flatten_multi_scale_feats(feats_i)
            pts_embed_i = self.attn(
                query=scene_embed_i[:, pts_mask_i],  # 3D场景特征
                value=feat_flatten,  # 2D图像特征
                query_pos=scene_pos_i[:, pts_mask_i] if scene_pos is not None else None,
                ref_pts=ref_pix_i[:, pts_mask_i].unsqueeze(2).expand(-1, -1, len(feats_i), -1),
                spatial_shapes=shapes,
                level_start_index=get_level_start_index(shapes),
            )  # 用DCA计算三维场景中，位于物体表面的voxel的特征，忽略Empty和被遮挡的点，从而保证从图像中获取特征时，不会有错误投影的情况
            # out_i         # (1,64,60,60,36)
            out_i = index_fov_back_to_voxels(
                nlc_to_nchw(scene_embed_i, self.scene_shape), pts_embed_i, pts_mask_i
            )
            outs.append(out_i)
        if ret_mask:
            return torch.cat(outs, dim=0), torch.stack(masks, dim=0)
        return torch.cat(outs, dim=0)