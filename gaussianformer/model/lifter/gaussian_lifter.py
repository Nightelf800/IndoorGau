import torch, torch.nn as nn
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        # gaussian generation number
        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz)
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
        self.xyz = xyz
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        self.semantic_dim = semantic_dim
        # xyz=(0,1), scale=(0,1), rots, opacity=0.1, semantic=17
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        # (Properties)
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        # (Queries)instance_feature.shape=[25600, 128]
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        # 51200,12 -> 51200,1 opacity    sigmoid
        # 51200,12 -> 51200,3 scale      softplus
        # 51200,12 -> 51200,4 rot        Norm
        self.opacity_head = nn.Linear(self.semantic_dim, 1)
        self.scale_head = nn.Linear(self.semantic_dim, 3)
        self.rot_head = nn.Linear(self.semantic_dim, 4)
        


    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        # anchor.requires_grad_(True)
        # instance_feature.requires_grad_(True)
        # import pdb;
        # pdb.set_trace()

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


        # 33 kuailong  55genggui 