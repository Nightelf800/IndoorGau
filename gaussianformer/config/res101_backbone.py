# ========= model config ===============
embed_dims = 128
num_anchor = 25600
num_decoder = 4
num_single_frame_decoder = 1
pc_range = [-20, -20, -10, 20, 20, 10]
# pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'checkpoints/r50_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 12
num_levels = 4

model = dict(
    type="BEVSegmentor",
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # with_cp = True, # 存在checkpoint的问题
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/r50_dcn_fcos3d_pretrain.pth'),  # 添加预训练权重
        ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        # add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],),

)
