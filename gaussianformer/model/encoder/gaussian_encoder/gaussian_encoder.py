from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features
        anchor = representation

        # embedding layer
        anchor_embed = self.anchor_encoder(anchor)
        # print(f'anchor_embed.shape: {anchor_embed.shape}')

        prediction = []
        for i, op in enumerate(self.operation_order):
            # print(f'{i}: {op}')
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    metas['voxel_size'],
                    metas['voxel_origin'])
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    anchor_encoder=self.anchor_encoder,
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    metas['voxel_size'],
                    metas['voxel_origin'],
                )
                # print('gaussian.semantics.shape: {}'.format(gaussian.semantics.shape))
                # predicted_classes = torch.argmax(gaussian.semantics, dim=-1)
                # predicted_classes = predicted_classes.squeeze()
                # print(f'predicted_classes.min: {predicted_classes.min()}')
                # print(f'predicted_classes.max: {predicted_classes.max()}')
                # class_counts = torch.bincount(predicted_classes.type(torch.long), minlength=12)
                # print(f'>>> Gaussian each category: {class_counts}')
                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction}