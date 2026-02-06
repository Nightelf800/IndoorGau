import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
from ..encoders import dinov2_encoder
from .. import decoders
from ..decoders import SymphoniesDecoder
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss, \
    mae_loss, silog_loss, cosine_loss, hvm_ce_ssc_loss
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

class GaussianFormer(nn.Module):

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
        image_shape=(370, 1220),
        pc_range=[0, 0, 0, 4, 4, 2],
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        depth=None,
        render=False,
        use_hvm=False,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions
        self.gaussian_weight = 0.5
        self.symphonies_weight = 0.5
        self.voxel_size = voxel_size

        self.encoder = build_from_configs(
            encoders, encoder, in_channels=768, embed_dims=embed_dims)
        self.gaussian_decoder = build_from_configs(
            decoders, decoder, embed_dims=embed_dims, use_hvm=use_hvm
        )
        self.pc_range = pc_range
        self.render = render
        self.use_hvm = use_hvm

        # depth_eval
        self.depth_model = depth['depth_model']
        if depth['depth_model'] == 'depthanything':
            # self.depth_eval_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(depth_encoder)).eval()

            overwrite = {**kwargs, "pretrained_resource": depth['depth_pretrained_resource']} if depth['depth_pretrained_resource'] else kwargs
            config = get_config(depth['depth_model_name'], "eval", depth['depth_dataset'], **overwrite)
            self.depth_eval_model = build_model(config)

    def forward(self, inputs):
        if len(inputs['img'].shape) == 3:
            inputs['img'] = inputs['img'].unsqueeze(0)
        h, w = inputs['img'].shape[-2:]
       
        # print(f'-------inputs paras----------')
        # print(f'inputs.keys: {inputs.keys()}')
        # for key in inputs.keys():
        #     if isinstance(inputs[key], str):
        #         print(f'key: {key}, name: {inputs[key]}')
        #     elif isinstance(inputs[key], int):
        #         print(f'key: {key}, value: {inputs[key]}')
        #     elif isinstance(inputs[key], list):
        #         print(f'key: {key}, value: {inputs[key]}')
        #     else:
        #         print(f'key: {key}, shape: {inputs[key].shape}')

        # depth_eval
        # print('depth_eval: {}'.format(inputs['depth_eval']))
        # print('depth_model: {}'.format(self.depth_model))
        # print('inputs[img].shape: {}'.format(inputs['img'].shape))
        # if inputs['depth_eval'][-1]:
        #     if self.depth_model == 'depthanything':
        #         # depth_eval_image = self.depth_eval_transform({'image': inputs['img']})['image']
        #         focal = torch.Tensor([715.0873]).cuda()  # This magic number (focal) is only used for evaluating BTS model

        #         with torch.no_grad():

        #             depth = self.depth_infer(self.depth_eval_model, inputs['img'], dataset='nyu', focal=focal)

        #             # depth = self.depth_eval_model(inputs['img'])['metric_depth']
        #             # print(f'depth.shape: {depth.shape}')
        #             depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        #             pred_min = depth.min()
        #             pred_max = depth.max()
        #             depth = (depth - pred_min) / (pred_max - pred_min) * 5.1980
        #             inputs['depth'] = depth

        #         # print(f'depth_model: {self.depth_model}, depth.shape: {depth.shape}')

        #         from PIL import Image
                # pred_min = depth.min()
                # pred_max = depth.max()
                # print(f'pred.max: {pred_max}')
                # print(f'pred.min: {pred_min}')
                # depth = (depth - pred_min) / (pred_max - pred_min) * 255
                # p = depth.squeeze().cpu().numpy()
                # p_uint8 = np.clip(p, 0, 255).astype(np.uint8)
                # Image.fromarray(p_uint8).save(f"./visual/pred.png")



        # GauusianEncoder
        # ms_img_feats = self.encoder(inputs['img'].unsqueeze(1))

        # maskdino
        pred_insts = self.encoder(inputs['img'])
        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')

        ms_img_feats = []
        for i in range(len(feats)):
            ms_img_feats.append(feats[i].unsqueeze(1))

        if 'vox_tsdf' in inputs:
            pred_insts['vox_tsdf'] = inputs['vox_tsdf']

        # depth, K, E, voxel_origin, projected_pix, fov_mask = list(
        #     map(lambda k: inputs[k],
        #         ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
        #          f'fov_mask_{self.volume_scale}')))

        # 先注释
        # outs, symphoines_decoder_outs = self.decoder(pred_insts, feats, pred_masks, depth, K, E, voxel_origin, projected_pix,
        #             fov_mask)


        metas = {
            'img': inputs['img'].unsqueeze(1),
            'depth': inputs['depth'],
            'encoder_feat_ori': pred_insts['encoder_feat_ori'],
            'projection_mat': inputs['projection_mat'].to(torch.float32).unsqueeze(1),
            'image_wh': inputs['image_wh'],
            'voxel_size': self.voxel_size,
            'voxel_origin': inputs['voxel_origin'],
            'occ_xyz': inputs['xyz'],
            'occ_cam_mask': inputs[f'fov_mask_{self.volume_scale}'],
            'cam_K': inputs['cam_K'],
            'cam_pose': inputs['cam_pose'],
            'projected_pix_1': inputs['projected_pix_1'],
            'fov_mask_1': inputs['fov_mask_1'],
        }

        gaussian_deocder_outs = self.gaussian_decoder(metas=metas, ms_img_feats=ms_img_feats)

        # print(f'gaussian_deocder_outs.keys: {gaussian_deocder_outs.keys()}')

        if self.render:
            return {
                'rendered_feats': gaussian_deocder_outs['rendered_feats'], 
                'gt_feats': gaussian_deocder_outs['gt_feats'],
                'rendered_depth': gaussian_deocder_outs['rendered_depth'],
                'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
        else:
            if self.use_hvm:
                return {'aux_outputs': gaussian_deocder_outs['pred_occ'], 'ssc_logits': gaussian_deocder_outs['pred_occ'][-1], 'hvm_out_dict': gaussian_deocder_outs['hvm_out_dict']}
            else:
                return {'aux_outputs': gaussian_deocder_outs['pred_occ'], 'ssc_logits': gaussian_deocder_outs['pred_occ'][-1]}
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
        loss_map_extra = {
            'hvm_ce_ssc': hvm_ce_ssc_loss
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
                if 'rendered_depth' in preds:
                    losses['loss_' + loss + '_depth'] = loss_map[loss](preds['rendered_depth'], target['depth']) * scale
                # if 'rendered_feats' in preds:
                #     losses['loss_' + 'mae' + '_colors'] = loss_map['mae'](preds['rendered_feats'], target['img'])
            if 'rendered_feats' in preds:
                losses['loss_' + 'cosine' + '_feats'] = loss_map['cosine'](preds['rendered_feats'], preds['gt_feats'], weight=5.0)
        else:
            target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
            if 'aux_outputs' in preds:
                for i, pred in enumerate(preds['aux_outputs']):
                    scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                    for loss in self.criterions:
                        if loss not in loss_map:
                            continue
                        losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                            'ssc_logits': pred
                        }, target) * scale
            else:
                for loss in self.criterions:
                    if loss not in loss_map:
                        continue
                    losses['loss_' + loss] = loss_map[loss](preds, target)
            if 'hvm_out_dict' in preds:
                for loss in self.criterions:
                    if loss == 'hvm_ce_ssc':
                        losses['loss_' + loss] = loss_map_extra[loss](preds, target)

        return losses
