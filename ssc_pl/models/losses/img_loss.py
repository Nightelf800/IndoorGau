import torch
import torch.nn.functional as F


def mse_loss(pred, target):

    # print('pred[rendered_depth].shape: {}'.format(pred['rendered_depth'].shape))
    # print('target[depth].shape: {}'.format(target['depth'].shape))

    return F.mse_loss(
        pred.float(),
        target.float(),
    )


def ce_img_loss(pred, target):
    return F.cross_entropy(
        pred.float(),
        target.float(),
        ignore_index=255,
    )


def mae_loss(pred, target):
    pred = pred[target != 0]
    target = target[target != 0]
    return F.l1_loss(
        pred.flatten(),
        target.flatten(),
    )

def silog_loss(pred, target, eps=1e-6, weight=1.0):
    """
    计算 Scale-Invariant Logarithmic Loss (SILOG Loss)

    :param pred: 真实值 (Tensor)
    :param target: 预测值 (Tensor)
    :param epsilon: 避免对数计算中的零值 (float)
    :return: 计算得到的损失值 (float)
    """
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(
        pred.clamp(min=eps))

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(
        dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    return (weight * loss).mean()


def cosine_loss(pred, target, weight=1.0):
    """
    最原始的 Cosine Loss
    计算两个向量的余弦相似度，损失为1 - cosine_similarity
    """
    # 将特征展平为(batch_size * seq_len, feature_dim)格式
    pred_flat = pred.flatten(2).mT.flatten(0, 1)
    target_flat = target.flatten(2).mT.flatten(0, 1)
    
    
    loss = F.cosine_embedding_loss(pred_flat, target_flat, torch.ones_like(target_flat[:, 0]))
    
    return loss.mean() * weight
