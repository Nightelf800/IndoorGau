#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        # semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            # semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            # semantics
        )
        return logits

    @staticmethod # todo
    def backward(ctx, out_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            # semantics,
            out_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opacity_grad, cov3D_grad = _C.local_aggregate_backward(*args)
        # means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            # semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax

    def forward(
        self, 
        pts,
        means3D, 
        opacities, 
        # semantics, 
        scales, 
        cov3D,
        pc_min):

        assert not pts.requires_grad
        scales = scales.detach()


        points_int = ((pts - pc_min) / self.grid_size).to(torch.int)
        # print(f'points_int.min: {points_int.min()}')
        # print(f'points_int[:, 0].max: {points_int[:, 0].max()}')
        # print(f'points_int[:, 1].max: {points_int[:, 1].max()}')
        # print(f'points_int[:, 2].max: {points_int[:, 2].max()}')
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # print('model.head.pts', torch.isnan(pts).any())   # 检查 model.head.pts 是否有 NaN
        # print('model.head.pts', torch.isinf(pts).any())   # 检查 model.head.pts 是否有 Inf
        # print(f'pts.min: {pts.min()}')
        # print(f'pts.max: {pts.max()}')
        # print(f'pts.shape: {pts.shape}')
        # print('model.head.points_int', torch.isnan(points_int).any())   # 检查 model.head.points_int 是否有 NaN
        # print('model.head.points_int', torch.isinf(points_int).any())   # 检查 model.head.points_int 是否有 Inf
        # print(f'points_int.min: {points_int.min()}')
        # print(f'points_int.max: {points_int.max()}')
        # print(f'points_int.shape: {points_int.shape}')
        # print('model.head.means3D', torch.isnan(means3D).any())   # 检查 model.head.means3D 是否有 NaN
        # print('model.head.means3D', torch.isinf(means3D).any())   # 检查 model.head.means3D 是否有 Inf
        # print(f'means3D.min: {means3D.min()}')
        # print(f'means3D.max: {means3D.max()}')
        # print(f'means3D.shape: {means3D.shape}')
        # print('model.head.means3D_int', torch.isnan(means3D_int).any())   # 检查 model.head.means3D_int 是否有 NaN
        # print('model.head.means3D_int', torch.isinf(means3D_int).any())   # 检查 model.head.means3D_int 是否有 Inf
        # print(f'means3D_int.min: {means3D_int.min()}')
        # print(f'means3D_int.max: {means3D_int.max()}')
        # print(f'means3D_int.shape: {means3D_int.shape}')
        # print('model.head.opacities', torch.isnan(opacities).any())   # 检查 model.head.opacities 是否有 NaN
        # print('model.head.opacities', torch.isinf(opacities).any())   # 检查 model.head.opacities 是否有 Inf
        # print(f'opacities.min:  {opacities.min()}')
        # print(f'opacities.max:  {opacities.max()}')
        # print(f'opacities.shape: {opacities.shape}')
        # print('model.head.semantics', torch.isnan(semantics).any())   # 检查 model.head.semantics 是否有 NaN
        # print('model.head.semantics', torch.isinf(semantics).any())   # 检查 model.head.semantics 是否有 Inf
        # print(f'semantics.min:  {semantics.min()}')
        # print(f'semantics.max:  {semantics.max()}')
        # print(f'semantics.shape: {semantics.shape}')
        # print('model.head.radii', torch.isnan(radii).any())   # 检查 model.head.radii 是否有 NaN
        # print('model.head.radii', torch.isinf(radii).any())   # 检查 model.head.radii 是否有 Inf
        # print(f'radii.min:  {radii.min()}')
        # print(f'radii.max:  {radii.max()}')
        # print(f'radii.shape: {radii.shape}')
        # print('model.head.cov3D', torch.isnan(cov3D).any())   # 检查 model.head.cov3D 是否有 NaN
        # print('model.head.cov3D', torch.isinf(cov3D).any())   # 检查 model.head.cov3D 是否有 Inf
        # print(f'cov3D.min:  {cov3D.min()}')
        # print(f'cov3D.max:  {cov3D.max()}')
        # print(f'cov3D.shape: {cov3D.shape}')

        

        # Invoke C++/CUDA rasterization routine
        logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            # semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        # print('model.head.logits', torch.isnan(logits).any())   # 检查 model.head.logits 是否有 NaN
        # print('model.head.logits', torch.isinf(logits).any())   # 检查 model.head.logits 是否有 Inf
        # print(f'logits.min: {logits.min()}')
        # print(f'logits.max: {logits.max()}')
        # print(f'logits.shape: {logits.shape}')
        
        if not self.inv_softmax:
            return logits # n, c
        else:
            assert False