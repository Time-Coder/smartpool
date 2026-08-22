import numpy as np
import torch


def time_mask(spec, max_mask_ratio=0.25, num_masks=2):
    n_t = spec.shape[1]
    max_len = max(1, int(n_t * max_mask_ratio))
    for _ in range(num_masks):
        mask_len = np.random.randint(1, max_len + 1)
        start = np.random.randint(0, max(1, n_t - mask_len))
        spec[:, start:start + mask_len] = 0.0
    return spec


def freq_mask(spec, max_mask_ratio=0.25, num_masks=2):
    n_f = spec.shape[0]
    max_len = max(1, int(n_f * max_mask_ratio))
    for _ in range(num_masks):
        mask_len = np.random.randint(1, max_len + 1)
        start = np.random.randint(0, max(1, n_f - mask_len))
        spec[start:start + mask_len, :] = 0.0
    return spec


def add_gaussian_noise(spec, std=0.05):
    return spec + np.random.normal(0, std, spec.shape).astype(spec.dtype)


def augment(spec):
    spec = spec.copy()
    if np.random.random() < 0.5:
        spec = time_mask(spec, max_mask_ratio=0.25, num_masks=2)
    if np.random.random() < 0.5:
        spec = freq_mask(spec, max_mask_ratio=0.25, num_masks=2)
    if np.random.random() < 0.3:
        spec = add_gaussian_noise(spec, std=0.05)
    return spec


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)