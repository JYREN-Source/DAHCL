import math
from torch.optim import AdamW


def set_lr(optim, lr_value):
    """Set learning rate for all param groups."""
    for pg in optim.param_groups:
        pg["lr"] = lr_value


def cosine_lr(epoch_idx, total_epochs, base_lr, warmup_ep, min_lr):
    """
    Cosine annealing with linear warmup.

    epoch_idx: 1-based epoch index
    """
    if epoch_idx <= warmup_ep:
        return base_lr * epoch_idx / max(1, warmup_ep)
    T = max(1, total_epochs - warmup_ep)
    t = min(max(0, epoch_idx - warmup_ep), T)
    cos_out = 0.5 * (1 + math.cos(math.pi * t / T))
    return min_lr + (base_lr - min_lr) * cos_out


def maybe_fused_adamw(params, lr, weight_decay, use_fused=True):
    """
    Create AdamW optimizer, optionally using fused=True if supported.
    """
    if use_fused:
        try:
            return AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            pass
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def get_lambda_mid(epoch, total_epochs,
                   base_lambda_mid=1.0,
                   final_lambda_mid=0.2,
                   start_decay_ratio=0.7):
    """
    Linearly decay lambda_mid from base_lambda_mid to final_lambda_mid:

    - First (start_decay_ratio * total_epochs) epochs: keep base_lambda_mid
    - After that: linear decay to final_lambda_mid
    """
    start_epoch = int(total_epochs * start_decay_ratio)
    if epoch <= start_epoch:
        return base_lambda_mid
    decay_epochs = max(1, total_epochs - start_epoch)
    t = min(max(0, epoch - start_epoch), decay_epochs)
    return base_lambda_mid + (final_lambda_mid - base_lambda_mid) * (t / decay_epochs)