from importlib import import_module

import torch
import torch.nn as nn
import sys
from mmengine.config import Config
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
import torch
from ..dinov2 import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14

class dinov2_encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 model_name,
                 checkpoint_path,
                 patch = 14,
                 embed_dims=256,       # 64/128
                 num_queries=100,      # 100
                 freeze=True,
                 use_clstoken = True):
        super().__init__()
        self.intermediate_layer_idx = {
            'dinov2_vits14': [5, 8, 11],
            'dinov2_vitb14': [5, 8, 11], 
            'dinov2_vitl14': [11, 17, 23], 
            'dinov2_vitg14': [19, 29, 39]
            }
        self.patch_size = patch
        self.use_clstoken = use_clstoken
        self.hidden_dims = in_channels  # 256

        self.out_index = self.intermediate_layer_idx[model_name]
        if model_name == "dinov2_vits14":
            self.model = dinov2_vits14(pretrained=False)
        elif model_name == "dinov2_vitb14":
            self.model = dinov2_vitb14(pretrained=False)
        elif model_name == "dinov2_vitl14":
            self.model = dinov2_vitl14(pretrained=False)
        elif model_name == "dinov2_vitg14":
            self.model = dinov2_vitg14(pretrained=False)
        else:
            print(f"error: {model_name} not exists!")
            sys.exit()

        if checkpoint_path is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device('cpu')),
                strict=True)  # otherwise all the processes will put the loaded weight on rank 0 and may lead to CUDA OOM

        self.query_embed = nn.Embedding(num_queries, embed_dims)   # instance query 均匀分布进行随机初始化
        self.pts_embed = nn.Embedding(num_queries, 2)              # instance pts        

        if freeze:  ### 默认不fine_tune dinov2
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True        
        self.projects = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Conv2d(
                in_channels=self.hidden_dims,
                out_channels=embed_dims,
                kernel_size=3,
                stride=1,
                padding=1)
        ])
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * self.hidden_dims, self.hidden_dims),
                        nn.GELU()))

    def forward(self, imgs):
        B, _, w, h = imgs.shape
        feature = self.model.get_intermediate_layers(imgs, n=self.out_index, return_class_token=self.use_clstoken)
        feats = []
        for i, x in enumerate(feature):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0].unsqueeze(0)
 
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], w // self.patch_size, h // self.patch_size))
            
            x = self.projects[i](x)

            feats.append(x)

        bs = feats[0].size(0)
        return dict(                                                   
            queries=self.query_embed.weight.repeat(bs, 1, 1),
            feats=feats,
            pred_pts=self.pts_embed.weight.repeat(bs, 1, 1).sigmoid())
