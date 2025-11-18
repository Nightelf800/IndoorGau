import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from ... import build_from_configs
from .. import encoders
from ..decoders import SymphoniesDecoder, SymphoniesDecoderMultiBS
from ..losses import ce_ssc_loss, frustum_proportion_loss, geo_scal_loss, sem_scal_loss
# from depth_eval.depth_anything.dpt import DepthAnything
from depth_eval.zoedepth.utils.config import get_config
from depth_eval.zoedepth.models.builder import build_model
import pickle

class Symphonies(nn.Module):

    def __init__(
        self,
        encoder,
        embed_dims,
        scene_size,
        view_scales,
        volume_scale,
        num_classes,
        num_layers=3,
        image_shape=(370, 1220),
        pc_range=[0, 0, 0, 0, 0, 0],
        voxel_size=0.2,
        downsample_z=2,
        class_weights=None,
        criterions=None,
        depth=None,
        **kwargs,
    ):
        super().__init__()
        self.volume_scale = volume_scale
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.criterions = criterions

        self.encoder = build_from_configs(
            encoders, encoder, embed_dims=embed_dims, scales=view_scales)
        self.decoder = SymphoniesDecoder(
            embed_dims,
            num_classes,
            num_layers=num_layers,
            num_levels=len(view_scales),
            scene_shape=scene_size,
            project_scale=volume_scale,
            image_shape=image_shape,
            voxel_size=voxel_size,
            pc_range = pc_range,
            downsample_z=downsample_z)

        # self.decoder = SymphoniesDecoderMultiBS(
        #     embed_dims,
        #     num_classes,
        #     num_layers=num_layers,
        #     num_levels=len(view_scales),
        #     scene_shape=scene_size,
        #     project_scale=volume_scale,
        #     image_shape=image_shape,
        #     voxel_size=voxel_size,
        #     pc_range = pc_range,
        #     downsample_z=downsample_z,
        # )

        # depth_eval
        self.depth_model = depth['depth_model']
        if depth['depth_model'] == 'depthanything':
            # self.depth_eval_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(depth_encoder)).eval()

            overwrite = {**kwargs, "pretrained_resource": depth['depth_pretrained_resource']} if depth['depth_pretrained_resource'] else kwargs
            config = get_config(depth['depth_model_name'], "eval", depth['depth_dataset'], **overwrite)
            self.depth_eval_model = build_model(config)

    def forward(self, inputs):
        if inputs['img'].dim() == 3:
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
        if inputs['use_depth_eval']:
            if self.depth_model == 'depthanything':
                # depth_eval_image = self.depth_eval_transform({'image': inputs['img']})['image']
                focal = torch.Tensor([715.0873]).cuda()  # This magic number (focal) is only used for evaluating BTS model

                with torch.no_grad():

                    depth = self.depth_infer(self.depth_eval_model, inputs['img'], dataset='nyu', focal=focal)

                    # depth = self.depth_eval_model(inputs['img'])['metric_depth']
                    # print(f'depth.shape: {depth.shape}')
                    depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
                    pred_min = depth.min()
                    pred_max = depth.max()
                    depth = (depth - pred_min) / (pred_max - pred_min) * 5.1980
                    inputs['depth'] = depth

        #         # print(f'depth_model: {self.depth_model}, depth.shape: {depth.shape}')

        #         from PIL import Image
        #         # pred_min = depth.min()
        #         # pred_max = depth.max()
        #         # print(f'pred.max: {pred_max}')
        #         # print(f'pred.min: {pred_min}')
        #         # depth = (depth - pred_min) / (pred_max - pred_min) * 255
        #         # p = depth.squeeze().cpu().numpy()
        #         # p_uint8 = np.clip(p, 0, 255).astype(np.uint8)
        #         # Image.fromarray(p_uint8).save(f"./visual/pred.png")




        pred_insts = self.encoder(inputs['img'])

        # print(f'-------pred insts----------')
        # for key in pred_insts.keys():
        #     if isinstance(pred_insts[key], str):
        #         print(f'key: {key}, name: {pred_insts[key]}')
        #     elif isinstance(pred_insts[key], int):
        #         print(f'key: {key}, value: {pred_insts[key]}')
        #     elif isinstance(pred_insts[key], list):
        #         print(f'list: ')
        #         for i in range(len(pred_insts[key])):
        #             print(f'key: {key}[{i}], shape: {pred_insts[key][i].shape}')
        #     else:
        #         print(f'key: {key}, shape: {pred_insts[key].shape}')



        pred_masks = pred_insts.pop('pred_masks', None)
        feats = pred_insts.pop('feats')

        depth, K, E, voxel_origin, projected_pix, fov_mask = list(
            map(lambda k: inputs[k],
                ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
                 f'fov_mask_{self.volume_scale}')))
        # import pdb;
        # pdb.set_trace()
        outs = self.decoder(
            pred_insts,
            feats,
            pred_masks,
            depth,
            K,
            E,
            voxel_origin,
            projected_pix,
            fov_mask
        )
        
        return {'ssc_logits': outs[-1], 'aux_outputs': outs}

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

    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss
        }

        # print(f'class_weights: {self.class_weights}')
        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])
        # print(f'class_weights_update: {self.class_weights}')
        losses = {}
        if 'aux_outputs' in preds:
            for i, pred in enumerate(preds['aux_outputs']):
                scale = 1 if i == len(preds['aux_outputs']) - 1 else 0.5
                for loss in self.criterions:
                    losses['loss_' + loss + '_' + str(i)] = loss_map[loss]({
                        'ssc_logits': pred
                    }, target) * scale
        else:
            for loss in self.criterions:
                losses['loss_' + loss] = 0
                # losses['loss_' + loss] = loss_map[loss](preds, target)
        return losses
