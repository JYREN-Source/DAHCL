# losses/phcl.py

import torch
import torch.nn.functional as F

def phcl_loss(
    emb_u_all,
    probs_u_all,
    high_mask,
    mid_mask,
    proxy_weight,
    temperature=0.07,
    lambda_mid=1.0,
    eps=1e-8,
):

    device = emb_u_all.device
    B_all, d = emb_u_all.shape
    K = proxy_weight.shape[0]

    w = F.normalize(proxy_weight, dim=1)  # (K, d)

    loss_high = torch.tensor(0.0, device=device)
    loss_mid = torch.tensor(0.0, device=device)

    if high_mask.any():
        z_h = F.normalize(emb_u_all[high_mask], dim=1)   # (B_h, d)
        p_h = probs_u_all[high_mask]                     # (B_h, K)
        y_hat_h = p_h.argmax(dim=1)                      # (B_h,)

        logits_sim = torch.matmul(z_h, w.t()) / temperature  # (B_h, K)
        loss_high = F.cross_entropy(logits_sim, y_hat_h)

    if mid_mask.any():
        z_m = F.normalize(emb_u_all[mid_mask], dim=1)    # (B_m, d)
        p_m = probs_u_all[mid_mask]                      # (B_m, K)

        B_m = z_m.size(0)
        losses_mid = []

        k_top = min(3, K)
        prob_floor = 1.0 / float(K)

        for i in range(B_m):
            z_i = z_m[i:i+1]       # (1, d)
            p_i = p_m[i]           # (K,)

            top_vals, top_idx = torch.topk(p_i, k=k_top, dim=0)   # (k_top,), (k_top,)

            mask_high = top_vals > prob_floor
            if mask_high.any():
                sel_vals = top_vals[mask_high]
                sel_idx = top_idx[mask_high]
            else:
                sel_vals = top_vals
                sel_idx = top_idx

            weight_sum = sel_vals.sum()
            if weight_sum.item() <= eps:
                alpha_i = torch.ones_like(sel_vals) / sel_vals.numel()
            else:
                alpha_i = sel_vals / (weight_sum + eps)           # (k_eff,)

            w_pos = w[sel_idx]                                    # (k_eff, d)
            w_tilde_i = (alpha_i.unsqueeze(1) * w_pos).sum(dim=0) # (d,)
            w_tilde_i = F.normalize(w_tilde_i, dim=0)

            neg_mask = torch.ones(K, dtype=torch.bool, device=device)
            neg_mask[sel_idx] = False
            if neg_mask.any():
                w_neg_i = w[neg_mask]                             # (K_neg, d)

                sim_pos = (z_i * w_tilde_i.unsqueeze(0)).sum(dim=1, keepdim=True) / temperature  # (1,1)
                sim_neg = torch.matmul(z_i, w_neg_i.t()) / temperature                           # (1,K_neg)
                logits_i = torch.cat([sim_pos, sim_neg], dim=1)   # (1, 1+K_neg)

                target_i = torch.zeros(1, dtype=torch.long, device=device)
                loss_i = F.cross_entropy(logits_i, target_i)
            else:

                sim_pos = (z_i * w_tilde_i.unsqueeze(0)).sum(dim=1) / temperature  # (1,)
                loss_i = 1.0 - sim_pos.mean()

            losses_mid.append(loss_i)

        if len(losses_mid) > 0:
            loss_mid = torch.stack(losses_mid).mean()

    return loss_high + lambda_mid * loss_mid