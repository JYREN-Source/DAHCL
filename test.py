import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from Model.ConvneXtV2 import Model
from Data.dataset import PUDataManager


plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

CLASS_NAMES = [
    "KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24",
    "KB27", "KI14", "KI16", "KI17", "KI18", "KI21"
]
NUM_CLASSES = len(CLASS_NAMES)

MAX_TSNE_SAMPLES = 10000
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

DEFAULT_MODEL_PATH = "best_train_total.pth"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_IN_CHANS = 1
NUM_DOMAINS = 3  # source, source1, source2

DOMAIN_MOD_EPS = 1

INFER_MODE = "best_single_dom"
LAMBDA_DOM = 0.5
PRINT_DOMAIN_PROBS = True


def compute_dpas_single_domain(model, device, loader):

    model.eval()
    with torch.no_grad():
        W_C = model.classifier.fc.weight  # [K, D]
        W_norm = F.normalize(W_C, dim=1)  # [K, D]

    d_sum, count = None, 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = model.feature_extractor(imgs)
            _, emb = model.classifier(feats, return_embed=True)

            emb_norm = F.normalize(emb, dim=1)  # [B, D]
            sim = emb_norm @ W_norm.t()  # [B, K]
            d_batch = sim.mean(dim=0)  # [K]

            d_sum = d_batch if d_sum is None else d_sum + d_batch
            count += 1

    return d_sum / max(count, 1)


def compute_all_dpas(model, device, src_l_loader, src_u1_loader, src_u2_loader, tgt_loader):

    print("\n[DPAS] Computing DPAS for all domains...")

    d_src = compute_dpas_single_domain(model, device, src_l_loader)
    d_u1 = compute_dpas_single_domain(model, device, src_u1_loader)
    d_u2 = compute_dpas_single_domain(model, device, src_u2_loader)
    d_tgt = compute_dpas_single_domain(model, device, tgt_loader)

    sim_src = F.cosine_similarity(d_tgt, d_src, dim=0)
    sim_u1 = F.cosine_similarity(d_tgt, d_u1, dim=0)
    sim_u2 = F.cosine_similarity(d_tgt, d_u2, dim=0)

    dpas_weights = F.softmax(torch.stack([sim_src, sim_u1, sim_u2]), dim=0)

    print("[DPAS] Similarities:")
    print(f"  sim(target, source ) = {sim_src.item():.4f}")
    print(f"  sim(target, source1) = {sim_u1.item():.4f}")
    print(f"  sim(target, source2) = {sim_u2.item():.4f}")
    print(f"[DPAS] Fusion weights (softmax): {dpas_weights.cpu().numpy()}")

    print("[DPAS] Inter-source similarities:")
    print(f"  cos(source, source1) = {F.cosine_similarity(d_src, d_u1, dim=0).item():.4f}")
    print(f"  cos(source, source2) = {F.cosine_similarity(d_src, d_u2, dim=0).item():.4f}")
    print(f"  cos(source1, source2) = {F.cosine_similarity(d_u1, d_u2, dim=0).item():.4f}")

    return d_src, d_u1, d_u2, dpas_weights

def test_target_domain(
        model_path=DEFAULT_MODEL_PATH,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        in_chans=DEFAULT_IN_CHANS,
        max_tsne_samples=MAX_TSNE_SAMPLES,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        num_classes=NUM_CLASSES,
        in_chans=in_chans,
        num_domains=NUM_DOMAINS,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    manager = PUDataManager(batch_size=batch_size, num_workers=num_workers)
    src_l_loader = manager.build_domain("source")
    src_u1_loader = manager.build_domain("source1")
    src_u2_loader = manager.build_domain("source2")
    tgt_loader = manager.build_domain("test")

    with torch.no_grad():
        W_C = model.classifier.fc.weight  # [K, D]
        b_C = model.classifier.fc.bias

        d_src, d_u1, d_u2, dpas_weights = compute_all_dpas(
            model, device, src_l_loader, src_u1_loader, src_u2_loader, tgt_loader
        )

        def modulated_weight(d):
            gamma_raw = model.domain_modulator(d)
            Gamma = 1.0 + DOMAIN_MOD_EPS * torch.tanh(gamma_raw)
            #Gamma = DOMAIN_MOD_EPS * torch.sigmoid(gamma_raw)
            return W_C * Gamma, Gamma

        W_src, Gamma_src = modulated_weight(d_src)
        W_u1, Gamma_u1 = modulated_weight(d_u1)
        W_u2, Gamma_u2 = modulated_weight(d_u2)

        print(f"\n[Domain Modulation] ||Gamma_src - 1||_F = {torch.norm(Gamma_src - 1).item():.8f}")
        print(f"[Domain Modulation] ||Gamma_u1 - 1||_F = {torch.norm(Gamma_u1 - 1).item():.8f}")
        print(f"[Domain Modulation] ||Gamma_u2 - 1||_F = {torch.norm(Gamma_u2 - 1).item():.8f}")

    print(f"\n[Evaluate] Target domain = test | INFER_MODE = {INFER_MODE}")

    all_preds, all_labels = [], []

    all_margins = []
    all_entropies = []

    with torch.no_grad():
        for seg, label in tgt_loader:
            seg = seg.to(device)
            label = label.to(device)

            feats = model.feature_extractor(seg)
            _, emb = model.classifier(feats, return_embed=True)

            domain_probs = dpas_weights.unsqueeze(0).expand(seg.size(0), -1)

            if PRINT_DOMAIN_PROBS:
                m = domain_probs.mean(dim=0).cpu().numpy()
                print(f"Domain probs mean: {m}")

            logits_src = F.linear(emb, W_src, b_C)
            logits_base = F.linear(emb, W_C, b_C)
            logits_u1 = F.linear(emb, W_u1, b_C)
            logits_u2 = F.linear(emb, W_u2, b_C)

            if INFER_MODE == "base_only":
                logits_final = logits_base

            elif INFER_MODE == "soft_triple":
                logits_final = (
                        domain_probs[:, 0:1] * logits_base +
                        domain_probs[:, 1:2] * logits_u1 +
                        domain_probs[:, 2:3] * logits_u2
                )

            elif INFER_MODE == "base_plus_best_dom":
                best_dom = domain_probs.argmax(dim=1)
                logits_final = logits_base.clone()

                mask = best_dom == 1
                logits_final[mask] = (1 - LAMBDA_DOM) * logits_base[mask] + LAMBDA_DOM * logits_u1[mask]

                mask = best_dom == 2
                logits_final[mask] = (1 - LAMBDA_DOM) * logits_base[mask] + LAMBDA_DOM * logits_u2[mask]

            elif INFER_MODE == "best_single_dom":
                best_dom = domain_probs.argmax(dim=1)
                logits_final = torch.zeros_like(logits_base)
                logits_final[best_dom == 0] = logits_base[best_dom == 0]
                logits_final[best_dom == 1] = logits_u1[best_dom == 1]
                logits_final[best_dom == 2] = logits_u2[best_dom == 2]

            else:
                raise ValueError(INFER_MODE)

            # ===== NEW: logit margin =====
            top2 = torch.topk(logits_final, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1])
            all_margins.append(margin.cpu())

            # ===== NEW: entropy =====
            probs = F.softmax(logits_final, dim=1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            all_entropies.append(entropy.cpu())

            preds = logits_final.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_margins = torch.cat(all_margins).numpy()
    all_entropies = torch.cat(all_entropies).numpy()

    acc = (all_preds == all_labels).mean()
    print(f"\nClosed-set Accuracy: {acc:.4f}")
    print(f"[Confidence] Mean logit margin   : {all_margins.mean():.4f}")
    print(f"[Confidence] Mean entropy        : {all_entropies.mean():.4f}")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Closed-set)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix_test_closed.png", dpi=300)
    plt.close()

    plot_tsne_four_domains(
        model, device,
        src_l_loader, src_u1_loader, src_u2_loader, tgt_loader,
        max_samples=max_tsne_samples
    )

    plot_tsne_by_categories(
        model, device,
        src_l_loader, src_u1_loader, src_u2_loader, tgt_loader,
        max_samples=max_tsne_samples
    )
# ==========================================================
# ==========================================================

def plot_tsne_four_domains(
        model, device,
        src_l_loader, src_u1_loader, src_u2_loader, tgt_loader,
        max_samples
):
    model.eval()
    feats, tags = [], []

    with torch.no_grad():
        for loader, tag in [
            (src_l_loader, 0),
            (src_u1_loader, 1),
            (src_u2_loader, 1),
            (tgt_loader, 2),
        ]:
            for seg, _ in loader:
                seg = seg.to(device)
                feats_i = model.feature_extractor(seg)
                _, emb = model.classifier(feats_i, return_embed=True)
                feats.append(emb.cpu().numpy())
                tags.append(np.full(len(emb), tag))

    feats = np.concatenate(feats)
    tags = np.concatenate(tags)

    if len(feats) > max_samples:
        idx = np.random.choice(len(feats), max_samples, replace=False)
        feats, tags = feats[idx], tags[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_N_ITER,
        init="pca",
        random_state=TSNE_RANDOM_STATE
    )
    feats_2d = tsne.fit_transform(feats)
    """
    colors = {0: "#BD0404", 1: "#bdbdbd", 2: "#0000ff"}#0000ff
    labels = {0: "Labeled Source", 1: "Unlabeled Sources", 2: "Target"}
    """
    colors = {0: "#E7694F", 1: "#bdbdbd", 2: "#0103FC"} #F07040
    labels = {0: "Labeled Source", 1: "Unlabeled Sources", 2: "Target"}

    plt.figure(figsize=(8, 6))

    mask_u = (tags == 1)
    plt.scatter(
        feats_2d[mask_u, 0], feats_2d[mask_u, 1],
        s=10, alpha=0.5, c=colors[1],
        label=labels[1], zorder=1
    )

    mask_l = (tags == 0)
    plt.scatter(
        feats_2d[mask_l, 0], feats_2d[mask_l, 1],
        s=10, alpha=0.8, c=colors[0],
        label=labels[0], zorder=2
    )

    mask_t = (tags == 2)
    plt.scatter(
        feats_2d[mask_t, 0], feats_2d[mask_t, 1],
        s=10, alpha=0.5, c=colors[2],
        label=labels[2], zorder=3
    )

    plt.legend(fontsize=14)
    plt.title("t-SNE of 4 Domains (Embedding Space)")
    plt.tight_layout()
    plt.savefig("tsne_4domains_embedding.png", dpi=300)
    plt.close()

def plot_tsne_by_categories(
        model, device,
        src_l_loader, src_u1_loader, src_u2_loader, tgt_loader,
        max_samples
):

    model.eval()
    feats, cat_labels = [], []

    with torch.no_grad():
        for loader in [src_l_loader, src_u1_loader, src_u2_loader, tgt_loader]:
            for seg, label in loader:
                seg = seg.to(device)
                feats_i = model.feature_extractor(seg)
                _, emb = model.classifier(feats_i, return_embed=True)
                feats.append(emb.cpu().numpy())
                cat_labels.append(label.cpu().numpy())

    feats = np.concatenate(feats)
    cat_labels = np.concatenate(cat_labels).astype(int)

    if len(feats) > max_samples:
        idx = np.random.choice(len(feats), max_samples, replace=False)
        feats, cat_labels = feats[idx], cat_labels[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_N_ITER,
        init="pca",
        random_state=TSNE_RANDOM_STATE
    )
    feats_2d = tsne.fit_transform(feats)

    # 13类颜色
    palette = sns.color_palette("tab20", NUM_CLASSES)
    colors = sns.color_palette("viridis", NUM_CLASSES)

    plt.figure(figsize=(8, 6))

    for c in range(NUM_CLASSES):
        mask = (cat_labels == c)
        plt.scatter(
            feats_2d[mask, 0], feats_2d[mask, 1],
            s=10, alpha=0.5, c=colors[c],
            label=CLASS_NAMES[c],
            zorder=2
        )

    plt.legend(fontsize=10, ncol=2)
    plt.title("t-SNE of 13 Categories (Embedding Space)")
    plt.tight_layout()
    plt.savefig("tsne_13categories_embedding.png", dpi=300)
    plt.close()
# ==========================================================
# 入口
# ==========================================================

def get_args():
    parser = argparse.ArgumentParser(description="Test script with DPAS-based domain modulation")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--in-chans", type=int, default=DEFAULT_IN_CHANS)
    parser.add_argument("--max-tsne-samples", type=int, default=MAX_TSNE_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test_target_domain(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_chans=args.in_chans,
        max_tsne_samples=args.max_tsne_samples,
    )