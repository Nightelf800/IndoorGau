from importlib import import_module

import torch
import torch.nn as nn

from mmengine.config import Config
from mmseg.registry import MODELS


class GaussianFormerEncoder(nn.Module):
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

    def forward(self, imgs=None, metas=None, points=None, ms_img_feats=None):

        outs = self.extract_img_feat(imgs)
        # for i in range(len(outs)):
        #     print('outs[{}].shape: {}'.format(i, outs[i].shape))
        return outs

    def extract_img_feat(self, imgs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.model.img_backbone(imgs)
        # for i in range(len(img_feats_backbone)):
        #     print('img_feats_backbone[{}].shape: {}'.format(i, img_feats_backbone[i].shape))
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.model.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        # for i in range(len(img_feats)):
        #     print('img_feats[{}].shape: {}'.format(i, img_feats[i].shape))
        img_feats = self.model.img_neck(img_feats_backbone)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped