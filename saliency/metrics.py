"""Saliency metrics/losses (verbatim from the notebooks) plus a small helper."""
import torch


def normalize_map(s_map):
    """Normalizes the map so that the sum of all pixels is equal to 1 (probability distribution)"""
    batch_size = s_map.size(0)
    s_map_flat = s_map.view(batch_size, -1)
    # Avoid division by zero by adding a small epsilon
    s_map_sum = torch.sum(s_map_flat, dim=1, keepdim=True) + 1e-7
    s_map_norm = s_map_flat / s_map_sum
    return s_map_norm.view_as(s_map)

def kld_loss(pred, gt):
    """Kullback-Leibler Divergence Loss"""
    pred = normalize_map(pred)
    gt = normalize_map(gt)
    eps = 1e-7
    # KLD formula: Sum(GT * log(GT / (Pred + eps)))
    loss = gt * torch.log(gt / (pred + eps) + eps)
    return torch.mean(torch.sum(loss.view(loss.size(0), -1), dim=1))

def cc_loss(pred, gt):
    """Pearson Correlation Coefficient Loss (expressed as 1 - CC to minimize it)"""
    pred_flat = pred.view(pred.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    # Compute the mean values
    pred_mean = torch.mean(pred_flat, dim=1, keepdim=True)
    gt_mean = torch.mean(gt_flat, dim=1, keepdim=True)

    # Center the values by subtracting the mean
    pred_center = pred_flat - pred_mean
    gt_center = gt_flat - gt_mean

    # Pearson correlation formula
    cc_num = torch.sum(pred_center * gt_center, dim=1)
    cc_den = torch.sqrt(torch.sum(pred_center ** 2, dim=1) * torch.sum(gt_center ** 2, dim=1) + 1e-7)

    cc = cc_num / cc_den
    return torch.mean(1 - cc) # (1 - CC) is minimized to maximize CC

def sim_loss(pred, gt):
    """Histogram Similarity Loss (expressed as 1 - SIM)"""
    pred = normalize_map(pred)
    gt = normalize_map(gt)

    pred_flat = pred.view(pred.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    # Compute the pixel-wise minimum between the two normalized distributions
    sim = torch.sum(torch.min(pred_flat, gt_flat), dim=1)
    return torch.mean(1 - sim)

def single_loss(pred, gt, w_kld=1.0, w_cc=1.0, w_sim=1.0):
    """Combines the three losses for a single branch"""
    return w_kld * kld_loss(pred, gt) + w_cc * cc_loss(pred, gt) + w_sim * sim_loss(pred, gt)

def total_loss(final_map, s1, s2, s3, gt):
    """Computes the total loss by applying Deep Supervision to all branches"""
    l_final = single_loss(final_map, gt)
    l_s1 = single_loss(s1, gt)
    l_s2 = single_loss(s2, gt)
    l_s3 = single_loss(s3, gt)

    # Sum all losses
    return l_final + l_s1 + l_s2 + l_s3


def compute_metrics(pred, gt) -> dict:
    """CC, KLD and SIM between a predicted and a ground-truth map, both (N,1,H,W).

    Values are averaged over the batch, as in the notebooks' MIT1003 evaluation.
    """
    with torch.no_grad():
        pred = pred.float()
        gt = gt.float()
        return {
            "cc": float(1 - cc_loss(pred, gt)),
            "kld": float(kld_loss(pred, gt)),
            "sim": float(1 - sim_loss(pred, gt)),
        }
