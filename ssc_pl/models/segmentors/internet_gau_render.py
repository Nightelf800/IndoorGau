import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
from .. import decoders
from vggt.vggt_infer import VGGTInfer
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss, lovasz_softmax_loss, \
    mae_loss, silog_loss, cosine_loss
# from depth_eval.depth_anything.dpt import DepthAnything
from depth_eval.zoedepth.utils.config import get_config
from depth_eval.zoedepth.models.builder import build_model
from cfg_module import ConfigManager
# from ...engine import LitModule
import lightning as L
from ... import build_from_configs, evaluation, models
from ..utils import (cumprod, flatten_fov_from_voxels, flatten_multi_scale_feats, generate_grid,
                     get_level_start_index, index_fov_back_to_voxels, interpolate_flatten,
                     nchw_to_nlc, nlc_to_nchw, pix2vox, vox2pix)
from torchvision import transforms

class InternetGauRender(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        scale_factor=1,
        vggt_path="./checkpoints/VGGT-1B",
        image_shape=(370, 1220),
        pc_range=[0, 0, 0, 4, 4, 2],
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        depth=None,
        render=False,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions
        self.voxel_size = voxel_size

        self.vggt_infer = VGGTInfer(checkpoint_path=vggt_path)

        self.encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims, scale_factor=scale_factor
        )
        self.gaussian_decoder = build_from_configs(
            decoders, decoder, embed_dims=embed_dims
        )
        self.render = render


    def forward(self, inputs):
        if len(inputs['img'].shape) == 3:
            inputs['img'] = inputs['img'].unsqueeze(0)
        b, c, h, w = inputs['img'].shape
    

        vggt_res = self.vggt_infer(inputs['img'])

        # print(f'vggt_res.keys: {vggt_res.keys()}')
        # print(f'vggt_res[depth].shape: {vggt_res["depth"].shape}')
        # print(f'vggt_res[extrinsics].shape: {vggt_res["extrinsics"].shape}')
        # print(f'vggt_res[intrinsics].shape: {vggt_res["intrinsics"].shape}')
        # print(f'vggt_res[point_map].shape: {vggt_res["point_map"].shape}')
        
        
        # maskdino
        pred_insts = self.encoder(inputs['img'])
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')

        ms_img_feats = []
        for i in range(len(feats)):
            ms_img_feats.append(feats[i].unsqueeze(1))

        if 'vox_tsdf' in inputs:
            pred_insts['vox_tsdf'] = inputs['vox_tsdf']

        M_intrinsic = torch.eye(4).repeat(b, 1, 1).to(vggt_res['intrinsics'].device)
        M_intrinsic[:, :3, :3] = vggt_res['intrinsics']
        projection_mat = torch.matmul(M_intrinsic, vggt_res['extrinsics'])


        metas = {
            'img': inputs['img'].unsqueeze(1),
            'depth': vggt_res['depth'],
            'encoder_feat_ori': pred_insts['encoder_feat_ori'],
            'projection_mat': projection_mat.unsqueeze(1),  
            'image_wh': inputs['image_wh'],
            'voxel_size': self.voxel_size,
            'voxel_origin': inputs['voxel_origin'],

            'cam_K': vggt_res['intrinsics'],
            'cam_pose': vggt_res['extrinsics'],

        }
        
        

        gaussian_deocder_outs = self.gaussian_decoder(metas=metas, ms_img_feats=ms_img_feats)

        # print(f'gaussian_deocder_outs.keys: {gaussian_deocder_outs.keys()}')

        if self.render:
            return {
                'rendered_feats': gaussian_deocder_outs['rendered_feats'], 
                'gt_feats': gaussian_deocder_outs['gt_feats'],
                'rendered_depth': gaussian_deocder_outs['rendered_depth'],
                'gt_depth': vggt_res['depth']}
        else:
            return {'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
        # return {'ssc_logits': outs[-1], 'aux_outputs': outs}




    def depth_infer(self, model, images, **kwargs):
        """Inference with flip augmentation"""

        # images.shape = N, C, H, W
        def get_depth_from_prediction(pred):
            if isinstance(pred, torch.Tensor):
                pred = pred  # pass
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            elif isinstance(pred, dict):
                pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
            else:
                raise NotImplementedError(f"Unknown output type {type(pred)}")
            return pred

        pred1 = model(images, **kwargs)
        pred1 = get_depth_from_prediction(pred1)

        pred2 = model(torch.flip(images, [3]), **kwargs)
        pred2 = get_depth_from_prediction(pred2)
        pred2 = torch.flip(pred2, [3])

        mean_pred = 0.5 * (pred1 + pred2)

        return mean_pred

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            assert(pred.shape == target.shape)
            target = target.to(pred.device)
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss,
            'mae': mae_loss,
            'silog': silog_loss,
            'cosine': cosine_loss,
            # 'lovasz': lovasz_softmax_loss
        }


        # print(f'class_weights: {self.class_weights}')
        
        # print(f'class_weights_update: {self.class_weights}')
        losses = {}

        # print(f'----------------IMG DEBUG----------------')
        # print(f"preds['rendered_depth'].mean(): {preds['rendered_depth'].mean()}")
        # print(f"preds['rendered_depth'].max(): {preds['rendered_depth'].max()}")
        # print(f"preds['rendered_depth'].min(): {preds['rendered_depth'].min()}")
        # print(f"target['depth'].mean(): {target['depth'].mean()}")
        # print(f"target['depth'].max(): {target['depth'].max()}")
        # print(f"target['depth'].min(): {target['depth'].min()}")
        # print(f"preds['rendered_colors'].mean(): {preds['rendered_feats'].mean()}")
        # print(f"preds['rendered_colors'].max(): {preds['rendered_feats'].max()}")
        # print(f"preds['rendered_colors'].min(): {preds['rendered_feats'].min()}")
        # print(f"target['gt_colors'].mean(): {target['img'].mean()}")
        # print(f"target['gt_colors'].max(): {target['img'].max()}")
        # print(f"target['gt_colors'].min(): {target['img'].min()}")
        # print(f'-----------------------------------------')



        if self.render:
            for loss in self.criterions:
                scale = 1 if loss != 'mae' else 0.2
                if 'rendered_depth' in preds and loss in ['silog', 'mae']:
                    losses['loss_' + loss + '_depth'] = loss_map[loss](preds['rendered_depth'], preds['gt_depth']) * scale
                # if 'rendered_feats' in preds:
                #     losses['loss_' + 'mae' + '_colors'] = loss_map['mae'](preds['rendered_feats'], target['img'])
                if 'rendered_feats' in preds and loss == 'cosine':
                    losses['loss_' + loss + '_feats'] = loss_map[loss](preds['rendered_feats'], preds['gt_feats'], weight=5.0)
        else:
            target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
            if 'aux_outputs' in preds:
                for i, pred in enumerate(preds['aux_outputs']):
                    scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                    for loss in self.criterions:
                        losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                            'ssc_logits': pred
                        }, target) * scale
            else:
                for loss in self.criterions:
                    losses['loss_' + loss] = loss_map[loss](preds, target)

        return losses
