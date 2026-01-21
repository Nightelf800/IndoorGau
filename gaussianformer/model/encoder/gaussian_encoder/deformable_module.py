from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init
import torch, torch.nn as nn
import numpy as np
from typing import List, Optional
from ...utils.safe_ops import safe_sigmoid
from ...utils.utils import get_rotation_matrix
from .utils import linear_relu_ln

try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None


@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_learnable_pts=0,
            fix_scale=None,
            pc_range=None,
            voxel_size=0.04,
            scale_range=None,
            phi_activation='sigmoid',
            xyz_coordinate='polar',
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.offsets = (0.5, 0.5, 0.5)
        self.scale_range = scale_range
        self.voxel_size = voxel_size
        self.offsets = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.phi_activation = phi_activation
        self.xyz_coordinate = xyz_coordinate

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
            self,
            anchor,
            instance_feature=None,
            voxel_size=0.04,
            voxel_origin=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])

        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (safe_sigmoid(self.learnable_fc(instance_feature)
                                            .reshape(bs, num_anchor, self.num_learnable_pts, 3)) - 0.5
                               ) * voxel_size
            scale = torch.cat([scale, learnable_scale], dim=-2)

        gs_scales = safe_sigmoid(anchor[..., None, 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales
        # print(f'key_points.shape: {key_points.shape}')
        rots = anchor[..., 6:10]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2)

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        # print(f'key_points.shape: {key_points.shape}')

        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(anchor[..., :3])
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(anchor[..., :2])
            z = torch.remainder(anchor[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError

        pc_real_range = []
        if voxel_origin is not None:
            # 修改为批处理友好的张量操作（不再使用 .item()）
            pc_real_range.append(self.pc_range[0] + voxel_origin[:, 0] + self.offsets[0] * self.voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[1] + voxel_origin[:, 1] + self.offsets[1] * self.voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[2] + voxel_origin[:, 2] + self.offsets[2] * self.voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[3] + voxel_origin[:, 0] + self.offsets[0] * self.voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[4] + voxel_origin[:, 1] + self.offsets[1] * self.voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[5] + voxel_origin[:, 2] + self.offsets[2] * self.voxel_size)  # [bs]

            # 将 pc_real_range 从列表转为张量，方便后续广播
            pc_real_range = torch.stack(pc_real_range, dim=1)  # [bs, 6]

        if self.xyz_coordinate == 'polar':
            # 使用广播机制处理批处理维度
            rrr = xyz[..., 0] * (pc_real_range[:, 3] - pc_real_range[:, 0]).unsqueeze(-1) + pc_real_range[:, 0].unsqueeze(-1)
            theta = xyz[..., 1] * (pc_real_range[:, 4] - pc_real_range[:, 1]).unsqueeze(-1) + pc_real_range[:, 1].unsqueeze(-1)
            phi = xyz[..., 2] * (pc_real_range[:, 5] - pc_real_range[:, 2]).unsqueeze(-1) + pc_real_range[:, 2].unsqueeze(-1)
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        else:
            # 笛卡尔坐标系下的批处理计算
            xxx = xyz[..., 0] * (pc_real_range[:, 3] - pc_real_range[:, 0]).unsqueeze(-1) + pc_real_range[:, 0].unsqueeze(-1)
            yyy = xyz[..., 1] * (pc_real_range[:, 4] - pc_real_range[:, 1]).unsqueeze(-1) + pc_real_range[:, 1].unsqueeze(-1)
            zzz = xyz[..., 2] * (pc_real_range[:, 5] - pc_real_range[:, 2]).unsqueeze(-1) + pc_real_range[:, 2].unsqueeze(-1)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1)  # [bs, ..., 3]

        key_points = key_points + xyz.unsqueeze(2)

        return key_points


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
            self,
            embed_dims: int = 256,
            num_groups: int = 8,
            num_levels: int = 4,
            num_cams: int = 6,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            kps_generator: dict = None,
            use_deformable_func=False,
            use_camera_embed=False,
            residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            # 与原来有修改 *linear_relu_ln(embed_dims, 1, 2, 12)
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
            self,
            instance_feature: torch.Tensor,
            anchor: torch.Tensor,
            anchor_embed: torch.Tensor,
            feature_maps: List[torch.Tensor],
            metas: dict,
            anchor_encoder=None,
            **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature, metas['voxel_size'], metas['voxel_origin'])
        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = []
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)
        # for i in range(len(feature_maps)):
        #     print('feature_maps[{}].shape: {}'.format(i, feature_maps[i].shape))

        for (
                temp_feature_maps,
                temp_metas,
                temp_key_points,
                temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            weights = self._get_weights(
                instance_feature, temp_anchor_embed, metas
            )
            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                        .contiguous()
                        .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                points_2d_1 = (
                    self.project_points(
                        temp_key_points,
                        temp_metas["projection_mat"],
                        temp_metas.get("image_wh"),
                    )
                )
                # print(f'points_2d_1.shape: {points_2d_1.shape}')
                points_2d_2 = points_2d_1.permute(0, 2, 3, 1, 4)
                # print(f'points_2d_2.shape: {points_2d_2.shape}')
                points_2d_3 = points_2d_2.reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                # print(f'points_2d_3.shape: {points_2d_3.shape}')
                # for i in range(len(temp_feature_maps)):
                #     print(f'temp_feature_maps[{i}].shape: {temp_feature_maps[i].shape}')
                # print(f'points_2d.shape: {points_2d.shape}')
                # print(f'weights.shape: {weights.shape}')
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d_3, weights
                )
                # print(f'temp_features_next.shape: {temp_features_next.shape}')
                temp_features_next = temp_features_next.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        # print('--------_get_weights--------')
        # print(f'instance_feature.shape: {instance_feature.shape}')
        # print(f'anchor_embed.shape: {anchor_embed.shape}')
        feature = instance_feature + anchor_embed
        # print(f'feature.shape: {feature.shape}')
        if self.camera_encoder is not None:
            temp_metas_mat = metas["projection_mat"][:, :, :3].reshape(
                bs, self.num_cams, -1
            )
            # print(f'temp_metas_mat.shape: {temp_metas_mat.shape}')

            camera_embed = self.camera_encoder(temp_metas_mat

                                               )
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
                .reshape(bs, num_anchor, -1, self.num_groups)
                .softmax(dim=-2)
                .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        # print(f'weights.shape: {weights.shape}')
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                    1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]
        # print(f'key_points.shape: {key_points.shape}')

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # print(f'projection_mat.shape: {projection_mat.shape}')
        # print(f'projection_mat.shape: {projection_mat[:, :, None, None].shape}')
        # print(f'pts_extend.shape: {pts_extend.shape}')
        # print(f'pts_extend.shape: {pts_extend[:, None, ..., None].shape}')
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        )
        # print(f'points_2d_no_squeeze.shape: {points_2d.shape}')
        points_2d = points_2d.squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        # print(f'points_2d_2.shape: {points_2d.shape}')
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
            feature_maps: List[torch.Tensor],
            key_points: torch.Tensor,
            projection_mat: torch.Tensor,
            image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
            self,
            features: torch.Tensor,
            weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features
