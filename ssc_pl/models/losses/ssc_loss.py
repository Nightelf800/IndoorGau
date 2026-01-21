import torch
import torch.nn.functional as F
from ..layers.hvm_head import voxel_sample


def ce_ssc_loss(pred, target):
    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target['target'].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
    )

def hvm_ce_ssc_loss(pred, target):
    refined_pred = pred['hvm_out_dict']["refined_pred"]
    sampled_voxel_coords = pred['hvm_out_dict']["sampled_voxel_coords"]
    target = target['target']

    # print('refined_pred.shape: ', refined_pred.shape)
    # print('sampled_voxel_coords.shape: ', sampled_voxel_coords.shape)
    # print('target.shape: ', target.shape)
    

    gt_voxels = voxel_sample(
        target.float().unsqueeze(1),
        sampled_voxel_coords,
        mode="nearest",
        align_corners=False
    ).squeeze_(1).long()
    # print('gt_voxels.shape: ', gt_voxels.shape)


    # lga_voxels = voxel_sample(
    #     lga.float().unsqueeze(1),
    #     sampled_voxel_coords,
    #     mode="nearest",
    #     align_corners=False
    # ).squeeze_(1).long()

    return F.cross_entropy(
        refined_pred,
        gt_voxels,
        ignore_index=255,
    )


def sem_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)

    target = target['target']
    mask = (target != 255)
    
    # 检查mask是否有True值，如果没有直接返回0
    if not mask.any():
        return 0
    
    target = target[mask]
    
    # 确保target不为空
    if target.numel() == 0:
        return 0
    
    loss, cnt = 0, 0
    num_classes = pred.shape[1]
    for i in range(0, num_classes):
        p = pred[:, i]
        p = p[mask]
        
        # 确保p和target形状匹配
        if p.shape != target.shape:
            continue
            
        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0

        # 检查completion_target是否有True值
        if torch.sum(completion_target) > 0:
            cnt += 1.0
            nominator = (p * completion_target).sum()
            
            # 计算precision
            p_sum = p.sum()
            if p_sum > 0:
                precision = nominator / p_sum
                # 确保precision在[0, 1]范围内
                precision = torch.clamp(precision, 0.0, 1.0)
                loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
            
            # 计算recall
            completion_target_sum = completion_target.sum()
            if completion_target_sum > 0:
                recall = nominator / completion_target_sum
                # 确保recall在[0, 1]范围内
                recall = torch.clamp(recall, 0.0, 1.0)
                loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
            
            # 计算specificity
            neg_completion_target = 1 - completion_target
            neg_completion_sum = neg_completion_target.sum()
            if neg_completion_sum > 0:
                # 预计算分子以避免重复计算
                neg_nominator = ((1 - p) * neg_completion_target).sum()
                specificity = neg_nominator / neg_completion_sum
                # 确保specificity在[0, 1]范围内
                specificity = torch.clamp(specificity, 0.0, 1.0)
                loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    
    if cnt == 0:
        return 0
    return loss / cnt


def geo_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target['target']
    mask = (target != 255)

    empty_probs = pred[:, 0]
    nonempty_probs = 1 - empty_probs
    
    # 确保mask不为空
    if not mask.any():
        return 0
        
    empty_probs = empty_probs[mask]
    nonempty_probs = nonempty_probs[mask]

    nonempty_target = target != 0
    nonempty_target = nonempty_target[mask].float()

    intersection = (nonempty_target * nonempty_probs).sum()
    loss = 0
    if nonempty_probs.sum() > 0:
        precision = intersection / nonempty_probs.sum()
        # 确保precision在[0, 1]范围内
        precision = torch.clamp(precision, 0.0, 1.0)
        loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
    if nonempty_target.sum() > 0:
        recall = intersection / nonempty_target.sum()
        # 确保recall在[0, 1]范围内
        recall = torch.clamp(recall, 0.0, 1.0)
        loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
    if (1 - nonempty_target).sum() > 0:
        specificity = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        # 确保specificity在[0, 1]范围内
        specificity = torch.clamp(specificity, 0.0, 1.0)
        loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    return loss
    # return (F.binary_cross_entropy(precision, torch.ones_like(precision)) +
    #         F.binary_cross_entropy(recall, torch.ones_like(recall)) +
    #         F.binary_cross_entropy(specificity, torch.ones_like(specificity)))


def frustum_proportion_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)

    frustums_masks = target['frustums_masks']
    frustums_class_dists = target['frustums_class_dists']
    num_frustums = frustums_class_dists.shape[1]
    batch_cnt = frustums_class_dists.sum(0)  # n_fstm, n_cls

    frustum_loss = 0
    frustum_nonempty = 0
    for f in range(num_frustums):
        frustum_mask = frustums_masks[:, f].unsqueeze(1)
        prob = frustum_mask * pred  # bs, n_cls, H, W, D
        prob = prob.flatten(2).transpose(0, 1)
        prob = prob.flatten(1)  # n_cls, bs * H * W * D
        cum_prob = prob.sum(dim=1)  # n_cls

        total_cnt = batch_cnt[f].sum()
        total_prob = prob.sum()
        if total_prob > 0 and total_cnt > 0:
            fp_target = batch_cnt[f] / total_cnt
            cum_prob = cum_prob / total_prob
            
            # 确保cum_prob在[0, 1]范围内
            cum_prob = torch.clamp(cum_prob, 1e-7, 1.0)  # 使用1e-7而不是0来避免log(0)的问题

            nonzeros = fp_target != 0
            nonzero_p = cum_prob[nonzeros]
            frustum_loss += F.kl_div(torch.log(nonzero_p), fp_target[nonzeros], reduction='sum')
            frustum_nonempty += 1
    if frustum_nonempty == 0:
        return 0
    return frustum_loss / frustum_nonempty
    
