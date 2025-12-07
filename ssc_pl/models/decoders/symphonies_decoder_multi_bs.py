import copy
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from ..projections import VoxelProposalLayerMultiBS
from ..layers import (ASPP, DeformableSqueezeAttention, DeformableTransformerLayer,
                      LearnableSqueezePositionalEncoding, TransformerLayer, Upsample, HardVoxelMiningHead, DDRBlock3D)
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox)


class SymphoniesLayer(nn.Module):
    def __init__(self, embed_dims, num_heads=8, num_levels=3, num_points=4, query_update=True):
        super().__init__()
        self.query_image_cross_defrom_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels, num_points)
        self.scene_query_cross_attn = TransformerLayer(embed_dims, num_heads, mlp_ratio=0)
        self.scene_self_deform_attn = DeformableTransformerLayer(
            embed_dims,
            num_heads,
            num_levels=1,
            num_points=num_points * 2,
            attn_layer=DeformableSqueezeAttention)
        self.query_update = query_update
        if query_update:
            self.query_scene_cross_deform_attn = DeformableTransformerLayer(
                embed_dims,
                num_heads,
                num_levels=1,
                num_points=num_points * 2,
                attn_layer=DeformableSqueezeAttention,
                mlp_ratio=0)
            self.query_self_attn = TransformerLayer(embed_dims, num_heads)

    def forward(self,
                scene_embed,
                inst_queries,
                feats,
                scene_pos=None,
                inst_pos=None,
                ref_2d=None,
                ref_3d=None,
                ref_vox=None,
                fov_mask=None):
        B = scene_embed.shape[0]
        output_scene_embed = []
        output_inst_query = []
        for i in range(B):
            scene_embed_i = scene_embed[i:i + 1, ...]
            inst_queries_i = inst_queries[i:i + 1, ...]
            feat_i = [x[i:i + 1, ...] for x in feats]
            scene_pos_i = scene_pos[i:i + 1, ...] if scene_pos is not None else None
            inst_pos_i = inst_pos[i:i + 1, ...] if inst_pos is not None else None
            ref_2d_i = ref_2d[i:i + 1, ...] if ref_2d is not None else None
            ref_3d_i = ref_3d[i:i + 1, ...] if ref_3d is not None else None
            ref_vox_i = ref_vox[i:i + 1, ...] if ref_vox is not None else None
            fov_mask_i = fov_mask[i:i + 1, ...] if fov_mask is not None else None
            # fov_mask = torch.ones_like(fov_mask, device=fov_mask.device)    # TODO
            fov_mask_i = torch.ones_like(fov_mask_i, device=fov_mask_i.device)  # TODO 如果直接使用fov_mask,可能会造成显存跳变。
            # torch.cuda.empty_cache()
            scene_embed_fov_i = flatten_fov_from_voxels(scene_embed_i, fov_mask_i)
            scene_pos_fov_i = flatten_fov_from_voxels(scene_pos_i,
                                                      fov_mask_i) if scene_pos_i is not None else None
            scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed_i])
            scene_level_index = get_level_start_index(scene_shape)
            feats_flatten, feat_shapes = flatten_multi_scale_feats(feat_i)
            feats_level_index = get_level_start_index(feat_shapes)
            inst_queries_i = self.query_image_cross_defrom_attn(
                inst_queries_i,
                feats_flatten,
                query_pos=inst_pos_i,
                ref_pts=ref_2d_i,
                spatial_shapes=feat_shapes,
                level_start_index=feats_level_index)
            scene_embed_fov_i = self.scene_query_cross_attn(
                scene_embed_fov_i,
                inst_queries_i,
                inst_queries_i,
                scene_pos_fov_i,
                inst_pos_i
            )
            scene_embed_fov_i = self.scene_self_deform_attn(
                scene_embed_fov_i,
                scene_embed_flatten,
                query_pos=scene_pos_fov_i,
                ref_pts=torch.flip(ref_vox_i[:, fov_mask_i.squeeze()], dims=[-1]),  # TODO: assert bs == 1
                spatial_shapes=scene_shape,
                level_start_index=scene_level_index)
            scene_embed_i = index_fov_back_to_voxels(scene_embed_i, scene_embed_fov_i, fov_mask_i)
            scene_embed_flatten, scene_shape = flatten_multi_scale_feats([scene_embed_i])
            if not self.query_update:
                output_scene_embed.append(scene_embed_i)
                output_inst_query.append(inst_queries_i)
                continue
            # 更新 instance query
            inst_queries_i = self.query_scene_cross_deform_attn(
                inst_queries_i,
                scene_embed_flatten,
                query_pos=inst_pos_i,
                ref_pts=torch.flip(ref_3d_i, dims=[-1]),
                spatial_shapes=scene_shape,
                level_start_index=scene_level_index
            )
            inst_queries_i = self.query_self_attn(inst_queries_i, query_pos=inst_pos_i)
            output_scene_embed.append(scene_embed_i)
            output_inst_query.append(inst_queries_i)
        scene_embed = torch.cat(output_scene_embed, dim=0)
        inst_queries = torch.cat(output_inst_query, dim=0)
        return scene_embed, inst_queries


class SymphoniesDecoderMultiBS(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_layers,
                 num_levels,
                 scene_shape,
                 project_scale,
                 image_shape,
                 voxel_size=0.2,
                 pc_range = [-2, 0, -1, 2, 4, 1],
                 downsample_z=1,
                 use_tsdf=False,
                 use_hvm=False
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        scene_shape = [s // project_scale for s in scene_shape]
        if downsample_z != 1:
            self.ori_scene_shape = copy.copy(scene_shape)
            scene_shape[-1] //= downsample_z
        self.scene_shape = scene_shape
        self.num_queries = cumprod(scene_shape)
        self.image_shape = image_shape
        self.voxel_size = voxel_size * project_scale
        self.pc_range = pc_range
        self.downsample_z = downsample_z
        self.voxel_proposal = VoxelProposalLayerMultiBS(embed_dims, scene_shape)
        self.layers = nn.ModuleList([
            SymphoniesLayer(
                embed_dims,
                num_levels=num_levels,
                query_update=True if i != num_layers - 1 else False) for i in range(num_layers)
        ])
        self.scene_embed = nn.Embedding(self.num_queries, embed_dims)
        # NYUv2 crop:  voxel_size=0.04 [100,100,50] => [20,20,50]
        # self.scene_pos = LearnableSqueezePositionalEncoding((20, 20, 50),embed_dims, squeeze_dims=(5, 5, 1))
        # NYUv2 crop:  voxel_size=0.8  [50,50,25] => [10,10,25]
        # self.scene_pos = LearnableSqueezePositionalEncoding((10, 10, 25), embed_dims, squeeze_dims=(5, 5, 1))
        # OccScanNet   voxel_size=0.8   [60,60,36] => [12,12,36]
        self.scene_pos = LearnableSqueezePositionalEncoding((scene_shape[0] // 5, scene_shape[1] // 5, scene_shape[2]),
                                                            embed_dims, squeeze_dims=(5, 5, 1))
        # self.scene_pos = LearnableSqueezePositionalEncoding((12, 12, 36), embed_dims, squeeze_dims=(5, 5, 1))
        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid)
        voxel_grid = generate_grid(scene_shape, normalize=True)
        self.register_buffer('voxel_grid', voxel_grid)
        self.aspp = ASPP(embed_dims, (1, 3))
        assert project_scale in (1, 2)
        self.cls_head = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose3d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=(1, 1, downsample_z),
                    padding=1,
                    output_padding=(0, 0, downsample_z - 1),
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(),
            ) if downsample_z != 1 else nn.Identity(),
            Upsample(embed_dims, embed_dims) if project_scale == 2 else nn.Identity(),
            nn.Conv3d(embed_dims, num_classes, kernel_size=1)
        )
        #######
        self.use_tsdf = use_tsdf
        if self.use_tsdf:
            pool_flag = False
            self.d1 = nn.Conv3d(1, embed_dims // 4, 3, stride=1, bias=True, padding=1)
            self.d2 = DDRBlock3D(embed_dims // 4, embed_dims // 2, embed_dims // 2, units=1, pool=pool_flag,
                                 residual=True, batch_norm=True, inst_norm=False)
            self.d_out = DDRBlock3D(embed_dims // 2, embed_dims, embed_dims, units=1, pool=pool_flag, residual=True,
                                    batch_norm=True, inst_norm=False)
            self.d_fuse = DDRBlock3D(embed_dims + embed_dims, embed_dims, embed_dims, units=1, pool=False,
                                     residual=True, batch_norm=True, inst_norm=False)
        #######
        self.use_hvm = use_hvm
        if self.use_hvm:
            self.hvm_head = HardVoxelMiningHead(
                in_channel=embed_dims + 12, embed_dims=embed_dims, num_classes=num_classes
            )
            self.hvm_head_pre = HardVoxelMiningHead(
                in_channel=embed_dims + 12, embed_dims=embed_dims, num_classes=num_classes
            )

    @autocast(dtype=torch.float32)
    def forward(self, pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
                fov_mask, vox_tsdf=None, x3ds=None):
        inst_queries = pred_insts['queries']  # bs, n, c
        inst_pos = pred_insts.get('query_pos', None)
        bs = inst_queries.shape[0]

        if self.downsample_z != 1:
            projected_pix = interpolate_flatten(
                projected_pix, self.ori_scene_shape, self.scene_shape, mode='trilinear')
            fov_mask = interpolate_flatten(
                fov_mask, self.ori_scene_shape, self.scene_shape, mode='trilinear')
        if len(depth.shape) == 4:  # support occdepth
            depth = depth[:, 0, ...]
            K = K[:, 0, ...]
        vol_pts = pix2vox(
            self.image_grid,
            depth.unsqueeze(1),
            K,
            E,
            voxel_origin,
            self.voxel_size,
            downsample_z=self.downsample_z,
            pc_range = self.pc_range
        ).long()  # <Tensor, [B,H*W,3]> 输入图像每个像素对应的体素三维坐标(以voxel_origin为原点)

        ref_2d = pred_insts['pred_pts'].unsqueeze(2).expand(-1, -1, len(feats), -1)
        ref_3d = self.generate_vol_ref_pts_from_masks(
            pred_insts['pred_boxes'], pred_masks,
            vol_pts).unsqueeze(2) if pred_masks else self.generate_vol_ref_pts_from_pts(
            pred_insts['pred_pts'], vol_pts).unsqueeze(
            2)  # pred_insts['pred_pts']表示随机初始化的归一化的像素坐标，ref_3d: <Tensor, [B, num_queries=100, 1, 3]> 表示其对应的三维体素坐标（已归一化）
        ref_pix = (torch.flip(projected_pix, dims=[-1]) + 0.5) / torch.tensor(
            self.image_shape).to(projected_pix)
        ref_pix = torch.flip(ref_pix, dims=[-1])
        ref_vox = nchw_to_nlc(self.voxel_grid.unsqueeze(0)).unsqueeze(2).repeat(bs, 1, 1, 1)

        scene_embed = self.scene_embed.weight.repeat(bs, 1, 1)  # (1,129600,64)

        if x3ds is not None:
            x3ds = x3ds.permute(0, 1, 2, 4, 3).flatten(2).permute(0, 2, 1)  # [bs, xyz, c]
            scene_embed = scene_embed + x3ds

        scene_pos = self.scene_pos().repeat(bs, 1, 1)  # (1,129600,64)
        scene_embed = self.voxel_proposal(scene_embed, feats, scene_pos, vol_pts, ref_pix)

        if self.use_tsdf and vox_tsdf is not None:
            vox_tsdf = self.d1(vox_tsdf)
            vox_tsdf = self.d2(vox_tsdf)
            vox_tsdf = self.d_out(vox_tsdf)
            scene_embed = self.d_fuse(torch.cat([scene_embed, vox_tsdf], dim=1))  # cat

        scene_pos = nlc_to_nchw(scene_pos, self.scene_shape)  # (1,129600,64) -> (1,64,60,60,36)

        outs = []
        if self.use_hvm:
            hvm_outs_list = []
        for i, layer in enumerate(self.layers):
            scene_embed, inst_queries = layer(scene_embed, inst_queries, feats, scene_pos, inst_pos,
                                              ref_2d, ref_3d, ref_vox, fov_mask)
            if i == 2:
                scene_embed = self.aspp(scene_embed)
            if self.training or i == len(self.layers) - 1:
                outs.append(self.cls_head(scene_embed))

            if self.use_hvm:
                # extract last two layer voxel feats for voxel mining
                if i == len(self.layers) - 1 or i == len(self.layers) - 2:
                    hvm_outs_list.append(scene_embed)

        if self.use_hvm:
            hvm_out_dict = self.hvm_head(None, hvm_outs_list[-1])
            hvm_out_dict_pre = self.hvm_head_pre(None, hvm_outs_list[-2])

            return outs, hvm_out_dict, hvm_out_dict_pre, hvm_outs_list

        return outs

    def generate_vol_ref_pts_from_masks(self, pred_boxes, pred_masks, vol_pts):
        pred_boxes *= torch.tensor((self.image_shape + self.image_shape)[::-1]).to(pred_boxes)
        pred_pts = pred_boxes[..., :2].int()
        cx, cy, w, h = pred_boxes.split((1, 1, 1, 1), dim=-1)
        pred_boxes = torch.cat([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)],
                               dim=-1).int()
        pred_boxes[0::2] = pred_boxes[0::2].clamp(0, self.image_shape[1] - 1)
        pred_boxes[1::2] = pred_boxes[1::2].clamp(1, self.image_shape[1] - 1)

        pred_masks = F.interpolate(
            pred_masks.float(), self.image_shape, mode='bilinear').to(pred_masks.dtype)
        bs, n = pred_masks.shape[:2]
        for b, i in product(range(bs), range(n)):
            if pred_masks[b, i].sum().item() != 0:
                continue
            boxes = pred_boxes[b, i]
            pred_masks[b, i, boxes[1]:boxes[3], boxes[0]:boxes[2]] = True
            if pred_masks[b, i].sum().item() != 0:
                continue
            pred_masks[b, i, pred_pts[b, i, 1], pred_pts[b, i, 0]] = True
        pred_masks = pred_masks.flatten(2).unsqueeze(-1).to(vol_pts)  # bs, n, hw, 1
        vol_pts = vol_pts.unsqueeze(1) * pred_masks  # bs, n, hw, 3
        vol_pts = vol_pts.sum(dim=2) / pred_masks.sum(dim=2) / torch.tensor(
            self.scene_shape).to(vol_pts)
        return vol_pts.clamp(0, 1)

    def generate_vol_ref_pts_from_pts(self, pred_pts, vol_pts):
        # pred_pts    # (2,100,2)         # 预测点的像素坐标 （归一化0~1）
        # vol_pts     # (2,3072000,3)     # 图像像素对应的3D体素坐标
        pred_pts = pred_pts * torch.tensor(self.image_shape[::-1]).to(pred_pts)
        pred_pts = pred_pts.long()
        pred_pts = pred_pts[..., 1] * self.image_shape[1] + pred_pts[..., 0]  # [2,100]
        ###### 原版仅支持batch_size=1
        # assert pred_pts.size(0) == 1
        # ref_pts = vol_pts[:, pred_pts.squeeze()]
        # ref_pts = ref_pts / (torch.tensor(self.scene_shape) - 1).to(pred_pts)
        ###### 新支持多batch_size>=2
        pred_pts_expanded = pred_pts.unsqueeze(2).expand(-1, -1, vol_pts.size(2))  # [2,100] -> (2,100,3)
        ref_pts = torch.gather(input=vol_pts, dim=1, index=pred_pts_expanded)  # [2,3072000,3] & )[]
        ref_pts = ref_pts / (torch.tensor(self.scene_shape) - 1).to(pred_pts)
        return ref_pts.clamp(0, 1)