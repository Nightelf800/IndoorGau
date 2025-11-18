import torch
import torch.nn.functional as F


def mse_loss(pred, target):
    pred['rendered_depth'] = pred['rendered_depth'].permute(2, 0, 1)

    # print('pred[rendered_depth].shape: {}'.format(pred['rendered_depth'].shape))
    # print('target[depth].shape: {}'.format(target['depth'].shape))

    return F.mse_loss(
        pred['rendered_depth'].float(),
        target['depth'].float(),
    )


def ce_img_loss(pred, target):
    pred['rendered_seg'] = pred['rendered_seg'].permute(0, 3, 1, 2)

    return F.cross_entropy(
        pred['rendered_seg'].float(),
        target['seg_2d'].long(),
        ignore_index=255,
    )