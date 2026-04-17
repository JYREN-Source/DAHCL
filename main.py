import os
import csv
import math
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ConvneXtV2 import Model
from Data.dataset import PUDataManager
from losses.phcl import phcl_loss
from losses.grl import GradientReversal
from losses.proxy_reg import proxy_separation_loss
from utils.train_utils import (
    set_lr,
    cosine_lr,
    maybe_fused_adamw,
    get_lambda_mid,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="SSDG + DAHCL training script"
    )


    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--in-chans", type=int, default=1)
    parser.add_argument("--num-domains", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-path", type=str, default="best_train_total.pth")
    parser.add_argument("--save-after-steps", type=int, default=2000,
                        help="(Deprecated) Kept for backward compatibility, not used for best model selection")
    parser.add_argument("--best-window-ratio", type=float, default=0.05,
                        help="Fraction of last epochs used to select best model "
                             "(e.g., 0.01 for last 1%%, 0.02 for last 2%%)")
    parser.add_argument("--use-phcl", action="store_true", default=True,
                        help="Enable PHCL (proxy-based contrastive) loss")
    parser.add_argument("--tau1", type=float, default=0.6,
                        help="Lower confidence threshold upper bound")
    parser.add_argument("--tau2", type=float, default=0.9,
                        help="Mid / high confidence threshold upper bound")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Contrastive temperature")
    parser.add_argument("--lambda-phcl", type=float, default=0.5,
                        help="Weight of PHCL loss")

    parser.add_argument("--use-lambda-mid-decay", action="store_true", default=True,
                        help="Enable decay of mid-confidence weight over epochs")
    parser.add_argument("--lambda-mid-base", type=float, default=1.0,
                        help="Base weight of mid-confidence samples")
    parser.add_argument("--lambda-mid-final", type=float, default=0.1,
                        help="Final weight of mid-confidence samples")
    parser.add_argument("--lambda-mid-decay-ratio", type=float, default=0.7,
                        help="Portion of training after which lambda_mid starts decaying")
    parser.add_argument("--use-pl-ce", action="store_true", default=False,
                        help="Enable CE loss on high-confidence unlabeled samples")
    parser.add_argument("--lambda-pl", type=float, default=1.0,
                        help="Weight of pseudo-label CE loss (high-confidence only)")
    parser.add_argument("--use-proxy-reg", action="store_true", default=True,
                        help="Enable proxy separation regularization")
    parser.add_argument("--proxy-margin", type=float, default=0.2,
                        help="Max allowed cosine similarity between different class proxies")
    parser.add_argument("--lambda-proxy", type=float, default=1e-3,
                        help="Weight of proxy separation loss")
    parser.add_argument("--use-domain-mod", action="store_true", default=True,
                        help="Enable domain-conditioned classifier modulation for unlabeled source domains")
    parser.add_argument("--domain-ema-momentum", type=float, default=0.1,
                        help="EMA momentum for domain representation (0~1, larger = faster update)")
    parser.add_argument("--domain-mod-eps", type=float, default=1,
                        help="Residual strength eps in Gamma = 1 + eps * tanh(gamma_raw)")
    parser.add_argument("--domain-mod-warmup-ratio", type=float, default=0.3,
                        help="Fraction of total epochs to warm up domain modulation from 0 to 1 (0~1)")
    parser.add_argument("--mod-use-for-pl", action="store_true", default=True,
                        help="If True, use domain-modulated logits' confidence to filter pseudo labels / masks")
    parser.add_argument("--mod-use-for-phcl", action="store_true", default=False,
                        help="If True, also use domain-modulated probs/logits in PHCL and PL-CE")
    parser.add_argument("--lambda-plq", type=float, default=0.1,
                        help="Weight of DCR loss")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-domain", type=float, default=0.1,
                        help="Weight of DANN loss")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eta-min", type=float, default=3e-6,
                        help="Min LR for cosine annealing")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Enable mixed precision training")
    parser.add_argument("--use-fused-adamw", action="store_true", default=True,
                        help="Use fused AdamW if available")
    parser.add_argument("--grad-clip", type=float, default=5.0)

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    model = Model(
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        num_domains=args.num_domains,
    ).to(device)

    manager = PUDataManager(batch_size=args.batch_size)
    src_l_loader = manager.build_domain("source")
    src_u1_loader = manager.build_domain("source1")
    src_u2_loader = manager.build_domain("source2")

    optimizer = maybe_fused_adamw(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_fused=args.use_fused_adamw
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    grl = GradientReversal().to(device)

    epoch_ce_hist, epoch_domain_hist = [], []
    epoch_phcl_hist, epoch_pl_hist = [], []
    epoch_proxy_hist, epoch_total_hist = [], []
    epoch_lr_hist, epoch_lambda_mid_hist = [], []

    epoch_dpas_cos_hist = []
    epoch_gamma_diff_hist = []
    epoch_logits_diff_hist = []

    epoch_state_dicts = {}

    global_step = 0

    total_epochs = args.epochs
    window_ratio = max(0.0, min(1.0, args.best_window_ratio))
    window_size = max(1, int(total_epochs * window_ratio))
    start_epoch_for_best = total_epochs - window_size + 1

    d_u1_ema = None
    d_u2_ema = None
    ema_m = args.domain_ema_momentum

    print(
        f"Training: SSDG + PHCL={args.use_phcl} + "
        f"PL-CE={args.use_pl_ce} + ProxyReg={args.use_proxy_reg} + "
        f"DomainMod={args.use_domain_mod} | "
        f"mod_use_for_pl={args.mod_use_for_pl}, mod_use_for_phcl={args.mod_use_for_phcl} | "
        f"epochs={args.epochs}, batch_size={args.batch_size} | "
        f"best_window_ratio={window_ratio:.4f} (last {window_size} epochs)"
    )

    domain_mod_eps = args.domain_mod_eps
    domain_mod_warmup_ratio = args.domain_mod_warmup_ratio

    for epoch in range(1, args.epochs + 1):
        # 学习率
        lr_now = cosine_lr(
            epoch,
            args.epochs,
            args.lr,
            args.warmup_epochs,
            args.eta_min
        )
        set_lr(optimizer, lr_now)

        # PHCL 中置信权重调度
        if args.use_lambda_mid_decay:
            lambda_mid_now = get_lambda_mid(
                epoch=epoch,
                total_epochs=args.epochs,
                base_lambda_mid=args.lambda_mid_base,
                final_lambda_mid=args.lambda_mid_final,
                start_decay_ratio=args.lambda_mid_decay_ratio,
            )
        else:
            lambda_mid_now = args.lambda_mid_base
        epoch_lambda_mid_hist.append(lambda_mid_now)

        warmup_epochs = max(1, int(args.epochs * domain_mod_warmup_ratio))
        if epoch <= warmup_epochs:
            beta_domain_mod = epoch / float(warmup_epochs)
        else:
            beta_domain_mod = 1.0

        model.train()
        running_ce = running_domain = 0.0
        running_phcl = running_pl = running_proxy = running_total = 0.0

        running_dpas_cos = 0.0
        running_gamma_diff = 0.0
        running_logits_diff = 0.0
        diag_count = 0

        min_steps = min(len(src_l_loader), len(src_u1_loader), len(src_u2_loader))
        src_l_iter = iter(src_l_loader)
        src_u1_iter = iter(src_u1_loader)
        src_u2_iter = iter(src_u2_loader)

        for step in range(min_steps):
            src_l_imgs, src_l_labels = [t.to(device, non_blocking=True) for t in next(src_l_iter)]
            src_u1_imgs, _ = [t.to(device, non_blocking=True) for t in next(src_u1_iter)]
            src_u2_imgs, _ = [t.to(device, non_blocking=True) for t in next(src_u2_iter)]

            optimizer.zero_grad(set_to_none=True)

            p = float(epoch - 1) / args.epochs
            alpha = 2. / (1. + math.exp(-10 * p)) - 1.

            with torch.cuda.amp.autocast(enabled=args.use_amp):

                feats_l = model.feature_extractor(src_l_imgs)
                logits_l, _ = model.classifier(feats_l, return_embed=True)
                loss_ce = criterion_cls(logits_l, src_l_labels)

                feats_u1 = model.feature_extractor(src_u1_imgs)
                feats_u2 = model.feature_extractor(src_u2_imgs)

                logits_u1_base, emb_u1 = model.classifier(feats_u1, return_embed=True)
                logits_u2_base, emb_u2 = model.classifier(feats_u2, return_embed=True)

                if args.use_domain_mod:

                    W_C = model.classifier.fc.weight  # [K, D]

                    emb_u1_norm = F.normalize(emb_u1.detach(), dim=1)  # [B1, D]
                    emb_u2_norm = F.normalize(emb_u2.detach(), dim=1)  # [B2, D]
                    W_norm = F.normalize(W_C, dim=1)  # [K, D]

                    sim_u1 = emb_u1_norm @ W_norm.t()
                    sim_u2 = emb_u2_norm @ W_norm.t()

                    d_u1_batch = sim_u1.mean(dim=0)
                    d_u2_batch = sim_u2.mean(dim=0)

                    if d_u1_ema is None:
                        d_u1_ema = d_u1_batch.clone()
                    else:
                        d_u1_ema = ((1.0 - ema_m) * d_u1_ema + ema_m * d_u1_batch).detach()

                    if d_u2_ema is None:
                        d_u2_ema = d_u2_batch.clone()
                    else:
                        d_u2_ema = ((1.0 - ema_m) * d_u2_ema + ema_m * d_u2_batch).detach()

                    gamma_u1_raw = model.domain_modulator(d_u1_ema)  # [K, D]
                    gamma_u2_raw = model.domain_modulator(d_u2_ema)  # [K, D]

                    Gamma_u1 = 1.0 + domain_mod_eps * torch.tanh(gamma_u1_raw)
                    Gamma_u2 = 1.0 + domain_mod_eps * torch.tanh(gamma_u2_raw)


                    b_C = model.classifier.fc.bias

                    W_u1 = W_C * Gamma_u1
                    W_u2 = W_C * Gamma_u2

                    logits_u1_mod = F.linear(emb_u1, W_u1, b_C)
                    logits_u2_mod = F.linear(emb_u2, W_u2, b_C)

                    logits_u1 = (1.0 - beta_domain_mod) * logits_u1_base + beta_domain_mod * logits_u1_mod
                    logits_u2 = (1.0 - beta_domain_mod) * logits_u2_base + beta_domain_mod * logits_u2_mod
                else:
                    logits_u1 = logits_u1_base
                    logits_u2 = logits_u2_base


                emb_u_all = torch.cat([emb_u1, emb_u2], dim=0)

                logits_u_all_base = torch.cat([logits_u1_base, logits_u2_base], dim=0)
                probs_u_all_base = torch.softmax(logits_u_all_base, dim=-1)
                conf_base, _ = probs_u_all_base.max(dim=-1)

                logits_u_all_mod = torch.cat([logits_u1, logits_u2], dim=0)
                probs_u_all_mod = torch.softmax(logits_u_all_mod, dim=-1)
                conf_mod, _ = probs_u_all_mod.max(dim=-1)

                if args.mod_use_for_pl:
                    conf_for_mask = conf_mod  # torch.min(conf_base, conf_mod)
                else:
                    conf_for_mask = conf_base

                if args.mod_use_for_phcl:
                    probs_for_phcl_pl = probs_u_all_mod
                    logits_for_pl = logits_u_all_mod
                else:
                    probs_for_phcl_pl = probs_u_all_base
                    logits_for_pl = logits_u_all_base

                feats_cat = torch.cat([feats_l, feats_u1, feats_u2], dim=0)
                feats_rev = grl(feats_cat, alpha=alpha)
                domain_logits = model.domain_classifier(feats_rev)

                B_l = feats_l.size(0)
                B_u1 = feats_u1.size(0)
                B_u2 = feats_u2.size(0)

                dom_l = torch.zeros(B_l, dtype=torch.long, device=device)
                dom_u1 = torch.ones(B_u1, dtype=torch.long, device=device) * 1
                dom_u2 = torch.ones(B_u2, dtype=torch.long, device=device) * 2
                domain_labels = torch.cat([dom_l, dom_u1, dom_u2], dim=0)

                loss_domain = criterion_domain(domain_logits, domain_labels)

                proxy_weight = model.classifier.fc.weight  # (K, D)

                if args.use_phcl:
                    Q1 = torch.quantile(conf_for_mask.detach(), 0.25)
                    Q3 = torch.quantile(conf_for_mask.detach(), 0.75)
                    t_low = min(Q1.item(), args.tau1)
                    t_mid = min(Q3.item(), args.tau2)

                    high_mask = conf_for_mask >= t_mid
                    mid_mask = (conf_for_mask >= t_low) & (conf_for_mask < t_mid)

                    loss_phcl = phcl_loss(
                        emb_u_all=emb_u_all,
                        probs_u_all=probs_for_phcl_pl,
                        high_mask=high_mask,
                        mid_mask=mid_mask,
                        proxy_weight=proxy_weight,
                        temperature=args.temperature,
                        lambda_mid=lambda_mid_now,
                    )
                else:
                    high_mask = conf_for_mask >= args.tau2
                    loss_phcl = torch.tensor(0.0, device=device)

                if args.use_pl_ce and high_mask.any():
                    with torch.no_grad():
                        pseudo_labels = probs_for_phcl_pl.argmax(dim=-1)
                    loss_pl = criterion_cls(
                        logits_for_pl[high_mask],
                        pseudo_labels[high_mask]
                    )
                else:
                    loss_pl = torch.tensor(0.0, device=device)

                if args.use_domain_mod and high_mask.any():

                    emb_norm = F.normalize(emb_u_all, dim=1)  # [B, D]
                    proxy_norm = F.normalize(proxy_weight, dim=1)  # [K, D]

                    sim = emb_norm @ proxy_norm.t()

                    probs_base = probs_u_all_base
                    probs_mod = probs_u_all_mod

                    sim_h = sim[high_mask]  # [Bh, K]
                    probs_base_h = probs_base[high_mask]  # [Bh, K]
                    probs_mod_h = probs_mod[high_mask]  # [Bh, K]

                    Q_base = (probs_base_h * sim_h).sum(dim=1).detach()  #  detach
                    Q_mod = (probs_mod_h * sim_h).sum(dim=1)


                    loss_plq = F.relu(Q_base - Q_mod).mean()
                else:
                    loss_plq = torch.tensor(0.0, device=device)

                if args.use_proxy_reg:
                    loss_proxy = proxy_separation_loss(proxy_weight, margin=args.proxy_margin)
                else:
                    loss_proxy = torch.tensor(0.0, device=device)

                loss = (
                        loss_ce
                        + args.lambda_domain * loss_domain
                        + args.lambda_phcl * loss_phcl
                        + args.lambda_pl * loss_pl
                        + args.lambda_proxy * loss_proxy
                        + args.lambda_plq * loss_plq  # 新增
                )

            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_ce += loss_ce.item()
            running_domain += loss_domain.item()
            running_phcl += loss_phcl.item()
            running_pl += loss_pl.item()
            running_proxy += loss_proxy.item()
            running_total += loss.item()
            global_step += 1

            if args.use_domain_mod and d_u1_ema is not None and d_u2_ema is not None:
                running_dpas_cos += F.cosine_similarity(d_u1_ema, d_u2_ema, dim=0).item()
                running_gamma_diff += torch.norm(Gamma_u1 - Gamma_u2, p='fro').item()
                running_logits_diff += (logits_u1 - logits_u2).abs().mean().item()
                diag_count += 1

            del src_l_imgs, src_l_labels, src_u1_imgs, src_u2_imgs
            del feats_l, feats_u1, feats_u2, logits_l, logits_u1, logits_u2, domain_logits

        avg_ce = running_ce / max(1, min_steps)
        avg_domain = running_domain / max(1, min_steps)
        avg_phcl = running_phcl / max(1, min_steps)
        avg_pl = running_pl / max(1, min_steps)
        avg_proxy = running_proxy / max(1, min_steps)
        avg_total = running_total / max(1, min_steps)

        epoch_ce_hist.append(avg_ce)
        epoch_domain_hist.append(avg_domain)
        epoch_phcl_hist.append(avg_phcl)
        epoch_pl_hist.append(avg_pl)
        epoch_proxy_hist.append(avg_proxy)
        epoch_total_hist.append(avg_total)
        epoch_lr_hist.append(lr_now)

        if diag_count > 0:
            epoch_dpas_cos_hist.append(running_dpas_cos / diag_count)
            epoch_gamma_diff_hist.append(running_gamma_diff / diag_count)
            epoch_logits_diff_hist.append(running_logits_diff / diag_count)
        else:
            epoch_dpas_cos_hist.append(float('nan'))
            epoch_gamma_diff_hist.append(float('nan'))
            epoch_logits_diff_hist.append(float('nan'))

        if epoch >= start_epoch_for_best:
            epoch_state_dicts[epoch] = copy.deepcopy(model.state_dict())

        print(
            f"[Epoch {epoch}] step={global_step} | lr={lr_now:.2e} | "
            f"lambda_mid={lambda_mid_now:.3f} | "
            f"CE: {avg_ce:.4f}, DOMAIN: {avg_domain:.4f}, "
            f"PHCL: {avg_phcl:.4f}, PL: {avg_pl:.4f}, "
            f"ProxyReg: {avg_proxy:.4f}, TOTAL: {avg_total:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_total = float("inf")
    best_epoch = None
    best_state_dict = None

    for epoch in range(start_epoch_for_best, total_epochs + 1):
        loss_val = epoch_total_hist[epoch - 1]
        if loss_val < best_total:
            best_total = loss_val
            best_epoch = epoch
            best_state_dict = epoch_state_dicts.get(epoch)

    if best_state_dict is not None:
        torch.save(best_state_dict, args.save_path)
        print(
            f"Saved best model over last {window_ratio * 100:.2f}% epochs "
            f"(epoch {best_epoch}, total_loss={best_total:.4f}) → {args.save_path}"
        )
    else:
        print("No model saved (unexpected).")

    print(
        f"Training done. Best Train Total Loss in last "
        f"{window_ratio * 100:.2f}% epochs = {best_total:.4f}"
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "train_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr", "lambda_mid",
            "ce", "domain", "phcl", "pl", "proxy_reg", "total",
            "dpas_cos", "gamma_diff", "logits_diff"
        ])
        for i, (lr_i, lm_i, ce, dom, phcl, pl, prox, total, dcos, gdiff, ldiff) in enumerate(
                zip(epoch_lr_hist, epoch_lambda_mid_hist,
                    epoch_ce_hist, epoch_domain_hist,
                    epoch_phcl_hist, epoch_pl_hist,
                    epoch_proxy_hist, epoch_total_hist,
                    epoch_dpas_cos_hist, epoch_gamma_diff_hist, epoch_logits_diff_hist),
                start=1):
            writer.writerow([i, lr_i, lm_i, ce, dom, phcl, pl, prox, total, dcos, gdiff, ldiff])

    print(f"Saved training logs to {csv_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)