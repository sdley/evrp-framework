"""
Paper-ready training diagnostic figure.

Produces a clean 2×2 figure:
    Top row:    Health scores (early vs trained)
    Bottom row: Where learning happened (early vs trained)

Usage:
    python paper_diagnostic_figure.py \
        --early checkpoint_800.pt \
        --trained checkpoint_18k.pt \
        --init checkpoint_800.pt \
        --output paper_figures/training_diagnostic.pdf

    --early:   early-stage checkpoint
    --trained: later-stage or final checkpoint
    --init:    reference for drift computation (initialization or earliest checkpoint)
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# ANALYSIS (reused from model_diagnostic.py, stripped to essentials)
# ---------------------------------------------------------------------------

def analyze(state_dict):
    embed_w = state_dict["encoder.embed.weight"]
    input_dim = embed_w.shape[1]
    embed_dim = embed_w.shape[0]
    num_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith("encoder.layers")
    ))
    total_params = sum(p.numel() for p in state_dict.values())
    xavier_std = 1.0 / np.sqrt(embed_dim)

    # Per-layer attention stds
    layer_stds = []
    for i in range(num_layers):
        wq = state_dict[f"encoder.layers.{i}.attn.W_q.weight"]
        layer_stds.append(wq.std().item())

    # LayerNorm drift from init
    norm_drifts = []
    for i in range(num_layers):
        nw = state_dict[f"encoder.layers.{i}.norm1.weight"]
        nb = state_dict[f"encoder.layers.{i}.norm1.bias"]
        norm_drifts.append(abs(1.0 - nw.mean().item()))
        norm_drifts.append(abs(nb.mean().item()))

    # Feature embedding differentiation
    feat_stds = [embed_w[:, i].std().item() for i in range(input_dim)]

    # Decoder std
    dec_stds = []
    for mat in ["W_q", "W_k", "W_v"]:
        key = f"decoder.cross_attn.{mat}.weight"
        if key in state_dict:
            dec_stds.append(state_dict[key].std().item())

    return {
        "embed_dim": embed_dim,
        "input_dim": input_dim,
        "num_layers": num_layers,
        "total_params": total_params,
        "xavier_std": xavier_std,
        "layer_stds": layer_stds,
        "norm_drifts": norm_drifts,
        "feat_stds": feat_stds,
        "dec_stds": dec_stds,
    }


def compute_scores(state_dict, ref_dict=None):
    info = analyze(state_dict)
    scores = {}

    # 1. Encoder specialization: do layers differ from each other?
    layer_stds = info["layer_stds"]
    std_cv = np.std(layer_stds) / np.mean(layer_stds) if np.mean(layer_stds) > 0 else 0
    scores["Encoder\nspecialization"] = min(100, std_cv * 1000)

    # 2. LayerNorm adaptation
    avg_norm_drift = np.mean(info["norm_drifts"])
    scores["LayerNorm\nadaptation"] = min(100, avg_norm_drift * 5000)

    # 3. Feature differentiation
    feat_cv = np.std(info["feat_stds"]) / np.mean(info["feat_stds"]) if np.mean(info["feat_stds"]) > 0 else 0
    scores["Feature\ndifferentiation"] = min(100, feat_cv * 500)

    # 4. Decoder movement from init
    xavier = info["xavier_std"]
    avg_dec = np.mean(info["dec_stds"]) if info["dec_stds"] else xavier
    dec_div = abs(avg_dec - xavier) / xavier * 100
    scores["Decoder\nreadiness"] = min(100, dec_div * 10)

    # 5. Weight drift (only with reference)
    if ref_dict is not None:
        total_drift = sum(
            (state_dict[k] - ref_dict[k]).norm().item()
            for k in state_dict if k in ref_dict
        )
        normalized = total_drift / np.sqrt(info["total_params"])
        scores["Weight\ndrift"] = min(100, normalized * 50)

    return scores


def compute_component_drift(state_dict, ref_dict):
    drift = {}
    for prefix, label in [("encoder.embed", "Embedding"),
                          ("encoder.layers.0", "Encoder\nlayer 0"),
                          ("encoder.layers.1", "Encoder\nlayer 1"),
                          ("encoder.layers.2", "Encoder\nlayer 2"),
                          ("decoder", "Decoder"),
                          ("value_head", "Value\nhead")]:
        d = sum((state_dict[k] - ref_dict[k]).norm().item()
                for k in state_dict if k.startswith(prefix) and k in ref_dict)
        drift[label] = d
    return drift


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------

def make_figure(early_path, trained_path, init_path, output_path,
                early_label="Early (800 ep)", trained_label="Trained (18k ep)"):

    early_sd = torch.load(early_path, map_location="cpu", weights_only=False)
    trained_sd = torch.load(trained_path, map_location="cpu", weights_only=False)
    init_sd = torch.load(init_path, map_location="cpu", weights_only=False)

    # Compute scores
    early_scores = compute_scores(early_sd, init_sd)
    trained_scores = compute_scores(trained_sd, init_sd)

    # Compute drift
    early_drift = compute_component_drift(early_sd, init_sd)
    trained_drift = compute_component_drift(trained_sd, init_sd)

    # Compute overall
    early_vals = list(early_scores.values())
    trained_vals = list(trained_scores.values())
    early_overall = np.mean(early_vals)
    trained_overall = np.mean(trained_vals)

    # --- Color scheme ---
    c_good = "#27ae60"
    c_ok = "#f39c12"
    c_bad = "#c0392b"
    c_blue = "#2c3e50"
    c_light_blue = "#3498db"
    c_teal = "#16a085"
    c_light_gray = "#ecf0f1"

    def score_color(s):
        if s >= 60: return c_good
        if s >= 30: return c_ok
        return c_bad

    def verdict(s):
        if s >= 60: return "Ready"
        if s >= 30: return "Developing"
        return "Undertrained"

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5),
                              gridspec_kw={"hspace": 0.50, "wspace": 0.35})

    score_names = list(trained_scores.keys())

    # ---- Panel (a): Early health scores ----
    ax = axes[0, 0]
    vals = [early_scores[n] for n in score_names]
    y = np.arange(len(score_names))
    bar_colors = [score_color(v) for v in vals]

    bars = ax.barh(y, vals, color=bar_colors, alpha=0.85, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(score_names, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (0–100)", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, vals):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=9, fontweight="bold",
                color=score_color(val))

    ov_color = score_color(early_overall)
    ax.set_title(f"(a) {early_label}\nOverall: {early_overall:.0f}/100 — {verdict(early_overall)}",
                 fontsize=11, fontweight="bold", color=ov_color, loc="left")

    # ---- Panel (b): Trained health scores ----
    ax = axes[0, 1]
    vals = [trained_scores[n] for n in score_names]
    bar_colors = [score_color(v) for v in vals]

    bars = ax.barh(y, vals, color=bar_colors, alpha=0.85, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(score_names, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (0–100)", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, vals):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=9, fontweight="bold",
                color=score_color(val))

    ov_color = score_color(trained_overall)
    ax.set_title(f"(b) {trained_label}\nOverall: {trained_overall:.0f}/100 — {verdict(trained_overall)}",
                 fontsize=11, fontweight="bold", color=ov_color, loc="left")

    # ---- Panel (c): Early component drift ----
    ax = axes[1, 0]
    comp_names = list(early_drift.keys())
    comp_vals = list(early_drift.values())
    total = sum(comp_vals)
    comp_pcts = [v / total * 100 if total > 0 else 0 for v in comp_vals]

    x_pos = np.arange(len(comp_names))
    bars = ax.bar(x_pos, comp_pcts, color=c_light_blue, alpha=0.8,
                  edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp_names, fontsize=8)
    ax.set_ylabel("Share of total drift (%)", fontsize=7) #9
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    for bar, pct, val in zip(bars, comp_pcts, comp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, pct , #+ 1.2
                f"{pct:.0f}%", ha="center", fontsize=8, fontweight="bold",
                color=c_blue)

    # Highlight max
    max_idx = np.argmax(comp_pcts)
    bars[max_idx].set_color(c_teal)
    bars[max_idx].set_alpha(0.9)

    ax.set_title(f"(c) {early_label}: where learning happened",
                 fontsize=11, fontweight="bold", loc="left")

    # ---- Panel (d): Trained component drift ----
    ax = axes[1, 1]
    comp_vals = list(trained_drift.values())
    total = sum(comp_vals)
    comp_pcts = [v / total * 100 if total > 0 else 0 for v in comp_vals]

    bars = ax.bar(x_pos, comp_pcts, color=c_light_blue, alpha=0.8,
                  edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp_names, fontsize=8)
    ax.set_ylabel("Share of total drift (%)", fontsize=9)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    for bar, pct, val in zip(bars, comp_pcts, comp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, pct,
                f"{pct:.0f}%", ha="center", fontsize=8, fontweight="bold",
                color=c_blue)

    max_idx = np.argmax(comp_pcts)
    bars[max_idx].set_color(c_teal)
    bars[max_idx].set_alpha(0.9)

    ax.set_title(f"(d) {trained_label}: where learning happened",
                 fontsize=11, fontweight="bold", loc="left")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-ready training diagnostic figure")
    parser.add_argument("--early", required=True, help="Early checkpoint path")
    parser.add_argument("--trained", required=True, help="Later/trained checkpoint path")
    parser.add_argument("--init", required=True,
                        help="Reference checkpoint for drift (init or earliest)")
    parser.add_argument("--output", default="results/figures/training_diagnostic.pdf")
    parser.add_argument("--early-label", default="Early (800 ep)")
    parser.add_argument("--trained-label", default="Trained (18k ep)")
    args = parser.parse_args()

    make_figure(args.early, args.trained, args.init, args.output,
                args.early_label, args.trained_label)
