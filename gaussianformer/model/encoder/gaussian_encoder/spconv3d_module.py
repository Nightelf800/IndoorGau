from mmengine.registry import MODELS
from mmengine.model import BaseModule
import spconv.pytorch as spconv
import torch.nn as nn, torch
from functools import partial
from .utils import spherical2cartesian, cartesian


@MODELS.register_module()
class SparseConv3D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        phi_activation='loop',
        xyz_coordinate='polar',
        use_out_proj=False,
        kernel_size=5,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.layer = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        if xyz_coordinate == 'polar':
            self.get_xyz = partial(
                spherical2cartesian, phi_activation=phi_activation)
        else:
            self.get_xyz = partial(
                cartesian)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('offsets', torch.tensor((0.5, 0.5, 0.5), dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, instance_feature, anchor, voxel_size=0.04, voxel_origin=None):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        bs, g, _ = instance_feature.shape

        pc_real_range = []
        if voxel_origin is not None:
            # 修改为批处理友好的张量操作（不再使用 .item()）
            pc_real_range.append(self.pc_range[0] + voxel_origin[:, 0] + self.offsets[0] * voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[1] + voxel_origin[:, 1] + self.offsets[1] * voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[2] + voxel_origin[:, 2] + self.offsets[2] * voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[3] + voxel_origin[:, 0] + self.offsets[0] * voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[4] + voxel_origin[:, 1] + self.offsets[1] * voxel_size)  # [bs]
            pc_real_range.append(self.pc_range[5] + voxel_origin[:, 2] + self.offsets[2] * voxel_size)  # [bs]

            # 将 pc_real_range 从列表转为张量，方便后续广播
            pc_real_range = torch.stack(pc_real_range, dim=1)  # [bs, 6]

        # sparsify
        anchor_xyz = anchor[..., :3]
        anchor_xyz = self.get_xyz(anchor_xyz, pc_real_range).flatten(0, 1)

        if anchor_xyz.numel() == 0:
            return torch.zeros(bs, g, self.layer.out_channels, device=instance_feature.device)

        indices = anchor_xyz - anchor_xyz.min(0, keepdim=True)[0]
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)

        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, g, -1).flatten(0, 1),
            indices], dim=-1)

        spatial_shape = indices.max(0)[0] + 1  # 确保至少为 1

        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=spatial_shape,
            batch_size=bs)

        output = self.layer(input)
        output = output.features.unflatten(0, (bs, g))

        return self.output_proj(output)
