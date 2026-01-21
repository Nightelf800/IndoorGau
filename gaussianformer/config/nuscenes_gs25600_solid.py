_base_ = [
    './_base_/misc.py',
    './_base_/model.py',
    './_base_/surroundocc.py'
]

# =========== data config ==============
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
val_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
train_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
# =========== misc config ==============
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=2e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
# ========= model config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=17,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ])
        ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask"
)
# ========= model config ===============
embed_dims = 128
num_anchor = 25600
num_decoder = 4
num_single_frame_decoder = 1
# pc_range = [-2, 0, -1, 2, 4, 1]
pc_range = [0, 0, 0, 4, 4, 2]
voxel_size = 0.08
# pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.008, 0.05]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'checkpoints/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 12

model = dict(
    # img_backbone_out_indices=[0, 1, 2, 3],
    # img_backbone=dict(
    #     _delete_=True,
    #     type='ResNet',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN2d', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe',
    #     with_cp = True,
    #     dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    #     stage_with_dcn=(False, False, True, True),
    #     pretrained=load_from),
    # img_neck=dict(
    #     start_level=1),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                pc_range=pc_range,
                voxel_size=voxel_size,
                scale_range=scale_range,
                fix_scale=[
                    [0, 0, 0],
                    [0.5, 0, 0],
                    [-0.5, 0, 0],
                    [0, 0.5, 0],
                    [0, -0.5, 0],
                    [0, 0, 0.5],
                    [0, 0, -0.5],
                ],
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[0.2, 0.2, 0.1],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[voxel_size, voxel_size, voxel_size],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
        # operation_order=[
        #     "spconv",
        #     "norm",
        #     "deformable",
        #     "ffn",
        #     "norm",
        # ] * (num_decoder - num_single_frame_decoder) + [
        #     "spconv",
        #     "norm",
        #     "deformable",
        #     "ffn",
        #     "norm",
        #     "refine",
        # ] * num_single_frame_decoder,
        

    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=12,
        pc_range=pc_range,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,   # what means?
            H=int((pc_range[3] - pc_range[0]) / voxel_size),
            W=int((pc_range[4] - pc_range[1]) / voxel_size),
            D=int((pc_range[5] - pc_range[2]) / voxel_size),
            pc_min=[0, 0, 0],
            # pc_min=[-2, 0, -1],
            grid_size=voxel_size),
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[0, 0, 0, 4, 4, 2],
            voxel_size=voxel_size),
    )

)
