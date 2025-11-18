import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import MultiheadAttention
import torch.nn.functional as F

import os
import numpy as np

from mmseg.registry import MODELS
from .base_backbone import BaseBackbone

import open_clip


@MODELS.register_module()
class CLIPBackbone(BaseBackbone):
    def __init__(self,
                 model_name,
                 pretrained,
                 out_indices,
                 multi_scales,
                 **kwargs
                 ):
        super().__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.visual.output_tokens = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.out_indices = out_indices
        self.multi_scales = multi_scales

        self.process = transforms.Compose(
            [   
                transforms.Resize((224, 224), antialias=None),
            ]
        )

        # self.tile_sizes = torch.tensor([0.05, 0.25, 0.5])
        # self.strider_scaler_list = [np.interp(tr.item(), 
        #                                       [0.05, 0.15], 
        #                                       [1.0, 0.5]) for tr in self.tile_sizes]
        # self.embed_size = 512

    def forward(self, x):
        # with torch.no_grad():
        #     B, C, H, W = x.shape

        #     clip_embeds_list = []
        #     for i, tr in enumerate(self.tile_sizes):
                
        #         kernel_size = int(H * tr.item())
        #         stride = int(kernel_size * self.strider_scaler_list[i])
        #         padding = kernel_size // 2

        #         unfold_func = torch.nn.Unfold(
        #             kernel_size=kernel_size,
        #             stride=stride,
        #             padding=padding,
        #         )

        #         tiles = unfold_func(x).permute(2, 0, 1).reshape(-1, 3, kernel_size, kernel_size)

        #         clip_embeds = self.model.encode_image(self.process(tiles))
        #         clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        #         center_x = ((kernel_size - 1) / 2 - padding + stride * np.arange(np.floor((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)))
        #         center_y = ((kernel_size - 1) / 2 - padding + stride * np.arange(np.floor((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)))
        #         center_x = torch.from_numpy(center_x)
        #         center_y = torch.from_numpy(center_y)

        #         clip_embeds = clip_embeds.reshape((center_x.shape[0], center_y.shape[0], self.embed_size, -1))
        #         clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :, :]), dim=1)
        #         clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :, :]), dim=0)

        #         clip_embeds_list.append(clip_embeds.permute(3, 0, 1, 2))

        #     mix_feat = clip_embeds_list[0].clone().permute(0, 3, 1, 2).float()
        #     _, _, a, b = mix_feat.shape
        #     for i in range(1, len(self.tile_sizes)):
        #         feat = clip_embeds_list[i].permute(0, 3, 1, 2).float()
        #         feat_interp = F.interpolate(feat, size=(a, b), mode="nearest")
        #         mix_feat += feat_interp
        #     mix_feat = (mix_feat.permute(0, 2, 3, 1) / len(self.tile_sizes))
        #     clip = mix_feat.permute(0, 3, 1, 2)

        #     multi_x_list = []
        #     for scale in self.multi_scales:
        #         scaled_x = F.interpolate(clip, size=(scale[0], scale[1]), mode="bilinear", align_corners=False)
        #         multi_x_list.append(scaled_x)

        #     return multi_x_list

        with torch.no_grad():
            _, tokens = self.model.encode_image(self.process(x))
            B, P, C = tokens.shape
            H = W = int(P**0.5)  # = grid_size
            tokens = tokens.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

            x = F.interpolate(tokens, size=(self.multi_scales[0], self.multi_scales[1]), mode="bilinear", align_corners=False)
            # multi_x_list = []
            # for scale in self.multi_scales:
            #     scaled_x = F.interpolate(tokens, size=(scale[0], scale[1]), mode="bilinear", align_corners=False)
            #     multi_x_list.append(scaled_x)
            return tokens
            return multi_x_list
        
        
