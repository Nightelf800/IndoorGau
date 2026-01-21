import torch, torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from ssc_pl.models.utils.utils import pix2vox, generate_grid


@MODELS.register_module()
class GaussianLifterImg2vox(BaseLifter):
    def __init__(
        self,
        num_anchor,
        rand_anchor_porp,
        embed_dims,
        image_shape,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        voxel_scene=[50, 50, 25],
        pc_range=[0, 0, 0, 0, 0, 0],
    ):
        super().__init__()
        self.embed_dims = embed_dims

        rand_anchor = int(num_anchor * rand_anchor_porp)
        self.num_anchor = num_anchor
        self.img2vox_anchor = num_anchor - rand_anchor
        
        # gaussian generation number
        xyz = torch.rand(rand_anchor, 3, dtype=torch.float)
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz)
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
        self.xyz = xyz
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(rand_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((rand_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((rand_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(rand_anchor, semantic_dim, dtype=torch.float)
        self.semantic_dim = semantic_dim
        # xyz=(0,1), scale=(0,1), rots, opacity=0.1, semantic=17
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        # (Properties)
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        # (Queries)instance_feature.shape=[25600, 128]
        self.instance_feature = nn.Parameter(
            torch.zeros([self.num_anchor, self.embed_dims]),
            requires_grad=feat_grad,
        )

        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w


        self.register_buffer('image_grid', image_grid)
        self.pc_range = pc_range
        self.voxel_scene = voxel_scene

        hidden_dim = 64
        # 共享的特征提取层
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 分支预测不同属性
        self.scale_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.semantic_head = nn.Linear(hidden_dim, semantic_dim)


    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, metas=None, **kwargs):
        bs = ms_img_feats[0].shape[0]

        vol_pts = pix2vox(
            self.image_grid,
            metas['depth'].unsqueeze(1),
            metas['cam_K'],
            metas['cam_pose'],
            metas['voxel_origin'],
            metas['voxel_size'],
            downsample_z=1,
            pc_range=self.pc_range)

        # print(f'vol_pots.shape: {vol_pts.shape}')

        # vol_pts: [1, 307200, 3], 保持批次维度
        # 1. 筛选 x, y, z ∈ [0, 4] 的点
        mask = (vol_pts[..., 0] >= 0) & (vol_pts[..., 0] <= self.voxel_scene[0]) & \
               (vol_pts[..., 1] >= 0) & (vol_pts[..., 1] <= self.voxel_scene[1]) & \
               (vol_pts[..., 2] >= 0) & (vol_pts[..., 2] <= self.voxel_scene[2])  # [1, 307200]

        # 提取符合条件的点（保持批次维度）
        filtered_points = vol_pts[mask].reshape(bs, -1, 3)  # [1, N, 3], N是符合条件的点数
        # print(f'vol_pots_filtered.shape: {filtered_points.shape}')

        # 2. 随机选择 12800 个点
        N = filtered_points.shape[1]
        if N >= self.img2vox_anchor:
            # 随机选择 12800 个点（不重复）
            indices = torch.randperm(N)[:self.img2vox_anchor]  # [12800]
            sel_xyz = filtered_points[:, indices, :]  # [1, 12800, 3]
        else:
            repeat_times = (self.img2vox_anchor // N) + 1
            sel_xyz = filtered_points.repeat(1, repeat_times, 1)[:, :self.img2vox_anchor, :]  # [1, 12800, 3]

        # 最终结果 selected_points: [1, 12800, 3]
        # print(f'selected_points.shape: {sel_xyz.shape}')
        sel_pts = sel_xyz.reshape(-1, 3)  # [bs * 12800, 3]
        sel_pts_features = self.mlp(sel_pts)  # [bs * 12800, hidden_dim]
        # 预测各属性
        sel_pts_scale = torch.exp(self.scale_head(sel_pts_features))  # 保证scale为正
        sel_pts_rot = F.normalize(self.rot_head(sel_pts_features), dim=-1)  # 归一化四元数
        sel_pts_opacity = torch.sigmoid(self.opacity_head(sel_pts_features))  # [0, 1]
        sel_pts_semantic = self.semantic_head(sel_pts_features)  # 可加softmax如需概率

        # 恢复形状
        sel_scale = sel_pts_scale.reshape(bs, self.img2vox_anchor, 3)  # [bs, 12800, 3]
        sel_rot = sel_pts_rot.reshape(bs, self.img2vox_anchor, 4)  # [bs, 12800, 4]
        sel_opacity = sel_pts_opacity.reshape(bs, self.img2vox_anchor, 1)  # [bs, 12800, 1]
        sel_semantic = sel_pts_semantic.reshape(bs, self.img2vox_anchor, -1)  # [bs, 12800, semantic_dim]
        sel_anchor = torch.cat([sel_xyz, sel_scale, sel_rot, sel_opacity, sel_semantic], dim=-1)
        # print(f'sel_anchor.shape: {sel_anchor.shape}')



        instance_feature = torch.tile(
            self.instance_feature[None], (bs, 1, 1)
        )
        rand_anchor = torch.tile(self.anchor[None], (bs, 1, 1))
        anchor = torch.cat([rand_anchor, sel_anchor], dim=1)
        # print(f'anchor.shape: {anchor.shape}')
        # anchor.requires_grad_(True)
        # instance_feature.requires_grad_(True)
        # import pdb;
        # pdb.set_trace()

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


        # 33 kuailong  55genggui 