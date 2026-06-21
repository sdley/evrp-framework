"""
Model Training Health Diagnostic

Answers the question: "Has my model learned enough to produce meaningful solutions?"

This is part of the framework's interpretability layer. Unlike training curves
(which show reward over time), this diagnostic inspects the model's internal
state to detect:
    - Whether weights have moved meaningfully from initialization
    - Whether different layers have specialized
    - Whether the decoder can distinguish between nodes
    - Whether attention patterns are uniform or structured
    - Whether the value function has collapsed

Usage:
    python model_diagnostic.py checkpoint.pt
    python model_diagnostic.py checkpoint.pt --reference init_model.pt
    python model_diagnostic.py checkpoint.pt --output diagnostic.pdf

Can compare against:
    - A reference model (e.g., initialization or earlier checkpoint)
    - Expected initialization statistics (Xavier/Kaiming)
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import OrderedDict


# ---------------------------------------------------------------------------
# 1. EXTRACT DIAGNOSTICS FROM CHECKPOINT
# ---------------------------------------------------------------------------

def analyze_checkpoint(state_dict):
    """Extract all diagnostic metrics from a state_dict."""
    diag = {}

    # --- Architecture ---
    embed_w = state_dict["encoder.embed.weight"]
    diag["input_dim"] = embed_w.shape[1]
    diag["embed_dim"] = embed_w.shape[0]
    diag["num_layers"] = len(set(
        k.split(".")[2] for k in state_dict if k.startswith("encoder.layers")
    ))
    diag["total_params"] = sum(p.numel() for p in state_dict.values())

    # --- Per-component parameter counts ---
    diag["encoder_params"] = sum(
        p.numel() for k, p in state_dict.items() if k.startswith("encoder")
    )
    diag["decoder_params"] = sum(
        p.numel() for k, p in state_dict.items() if k.startswith("decoder")
    )
    diag["value_params"] = sum(
        p.numel() for k, p in state_dict.items() if k.startswith("value")
    )

    # --- Health checks ---
    diag["has_nan"] = any(torch.isnan(v).any().item() for v in state_dict.values())
    diag["has_inf"] = any(torch.isinf(v).any().item() for v in state_dict.values())

    # --- Per-layer attention weight statistics ---
    layer_stats = []
    for i in range(diag["num_layers"]):
        stats = {}
        for mat in ["W_q", "W_k", "W_v", "W_o"]:
            key = f"encoder.layers.{i}.attn.{mat}.weight"
            if key in state_dict:
                w = state_dict[key]
                stats[mat] = {"std": w.std().item(), "mean": w.mean().item(),
                              "min": w.min().item(), "max": w.max().item()}
        # FF network
        ff_key = f"encoder.layers.{i}.ff.0.weight"
        if ff_key in state_dict:
            w = state_dict[ff_key]
            stats["ff"] = {"std": w.std().item()}
        # LayerNorm
        norm_w_key = f"encoder.layers.{i}.norm1.weight"
        norm_b_key = f"encoder.layers.{i}.norm1.bias"
        if norm_w_key in state_dict:
            stats["norm_w_mean"] = state_dict[norm_w_key].mean().item()
            stats["norm_b_mean"] = state_dict[norm_b_key].mean().item()
            stats["norm_w_std"] = state_dict[norm_w_key].std().item()
            stats["norm_b_std"] = state_dict[norm_b_key].std().item()
        layer_stats.append(stats)
    diag["layer_stats"] = layer_stats

    # --- Decoder statistics ---
    dec_stats = {}
    for mat in ["W_q", "W_k", "W_v", "W_o"]:
        key = f"decoder.cross_attn.{mat}.weight"
        if key in state_dict:
            w = state_dict[key]
            dec_stats[mat] = {"std": w.std().item()}
    ctx_key = "decoder.proj_ctx.weight"
    if ctx_key in state_dict:
        dec_stats["proj_ctx"] = {"std": state_dict[ctx_key].std().item()}
    diag["decoder_stats"] = dec_stats

    # --- Value head statistics ---
    val_stats = {}
    for key in state_dict:
        if key.startswith("value_head"):
            w = state_dict[key]
            val_stats[key] = {"std": w.std().item(), "mean": w.mean().item()}
    diag["value_stats"] = val_stats

    # --- Per-feature embedding analysis ---
    feat_stats = []
    for feat_idx in range(diag["input_dim"]):
        col = embed_w[:, feat_idx]
        feat_stats.append({
            "mean": col.mean().item(),
            "std": col.std().item(),
            "min": col.min().item(),
            "max": col.max().item(),
        })
    diag["feature_stats"] = feat_stats

    return diag


def compute_drift(state_dict, ref_dict):
    """Compute weight drift between two checkpoints."""
    drift = {}

    # Total drift
    total = sum((state_dict[k] - ref_dict[k]).norm().item()
                for k in state_dict if k in ref_dict)
    drift["total"] = total

    # Per-component drift
    for prefix, label in [("encoder.embed", "embedding"),
                          ("encoder.layers", "encoder_layers"),
                          ("decoder", "decoder"),
                          ("value_head", "value_head")]:
        d = sum((state_dict[k] - ref_dict[k]).norm().item()
                for k in state_dict if k.startswith(prefix) and k in ref_dict)
        drift[label] = d

    # Per-layer drift
    num_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith("encoder.layers")
    ))
    layer_drifts = []
    for i in range(num_layers):
        prefix = f"encoder.layers.{i}"
        d = sum((state_dict[k] - ref_dict[k]).norm().item()
                for k in state_dict if k.startswith(prefix) and k in ref_dict)
        layer_drifts.append(d)
    drift["per_layer"] = layer_drifts

    # Per-feature embedding drift
    embed_key = "encoder.embed.weight"
    if embed_key in state_dict and embed_key in ref_dict:
        input_dim = state_dict[embed_key].shape[1]
        feat_drifts = []
        for feat_idx in range(input_dim):
            d = (state_dict[embed_key][:, feat_idx] -
                 ref_dict[embed_key][:, feat_idx]).norm().item()
            feat_drifts.append(d)
        drift["per_feature"] = feat_drifts

    # Per-component std change
    std_changes = {}
    for key in state_dict:
        if key in ref_dict and state_dict[key].dim() >= 2:
            old_std = ref_dict[key].std().item()
            new_std = state_dict[key].std().item()
            if old_std > 0:
                std_changes[key] = (new_std - old_std) / old_std * 100
    drift["std_changes"] = std_changes

    return drift


# ---------------------------------------------------------------------------
# 2. COMPUTE HEALTH SCORES
# ---------------------------------------------------------------------------

def compute_health_scores(diag, drift=None):
    """
    Compute interpretable health scores (0-100) for different aspects.
    These are heuristic but informative.
    """
    scores = {}

    # --- Encoder learning score ---
    # Based on how much layer stds have diverged from each other
    # (trained models have different std per layer; untrained are ~identical)
    layer_stds = [ls["W_q"]["std"] for ls in diag["layer_stats"]]
    std_variance = np.std(layer_stds) / np.mean(layer_stds) if np.mean(layer_stds) > 0 else 0
    # Scale: 0 = all identical (untrained), 100 = highly differentiated
    scores["encoder_specialization"] = min(100, std_variance * 1000)

    # --- Decoder readiness score ---
    # Based on whether decoder weights have moved from initialization range
    # Xavier init for (128, 128) gives std ≈ 0.0625
    xavier_std = 1.0 / np.sqrt(diag["embed_dim"])
    dec_stds = [v["std"] for v in diag["decoder_stats"].values()]
    avg_dec_std = np.mean(dec_stds) if dec_stds else xavier_std
    dec_divergence = abs(avg_dec_std - xavier_std) / xavier_std * 100
    scores["decoder_readiness"] = min(100, dec_divergence * 10)

    # --- LayerNorm drift score ---
    # Untrained: mean=1.0, bias=0.0. Trained: should diverge
    norm_drifts = []
    for ls in diag["layer_stats"]:
        if "norm_w_mean" in ls:
            norm_drifts.append(abs(1.0 - ls["norm_w_mean"]))
            norm_drifts.append(abs(ls["norm_b_mean"]))
    avg_norm_drift = np.mean(norm_drifts) if norm_drifts else 0
    scores["layernorm_adaptation"] = min(100, avg_norm_drift * 5000)

    # --- Feature differentiation score ---
    # Do different input features have different embedding patterns?
    feat_stds = [fs["std"] for fs in diag["feature_stats"]]
    feat_std_variance = np.std(feat_stds) / np.mean(feat_stds) if np.mean(feat_stds) > 0 else 0
    scores["feature_differentiation"] = min(100, feat_std_variance * 500)

    # --- Overall weight drift score (only if reference available) ---
    if drift is not None:
        # Normalize by total parameter count
        normalized_drift = drift["total"] / np.sqrt(diag["total_params"])
        scores["weight_drift"] = min(100, normalized_drift * 50)

        # Decoder vs encoder ratio (should be balanced in well-trained model)
        if drift["encoder_layers"] > 0:
            dec_enc_ratio = drift["decoder"] / drift["encoder_layers"]
            scores["decoder_encoder_balance"] = min(100, dec_enc_ratio * 100)
        else:
            scores["decoder_encoder_balance"] = 0
    else:
        scores["weight_drift"] = None
        scores["decoder_encoder_balance"] = None

    # --- Value function health ---
    val_stds = [v["std"] for v in diag["value_stats"].values()
                if "weight" in list(diag["value_stats"].keys())[0]]
    if val_stds:
        # Check if value head output is collapsing (std shrinking toward 0)
        output_key = [k for k in diag["value_stats"] if "2.weight" in k]
        if output_key:
            output_std = diag["value_stats"][output_key[0]]["std"]
            # Healthy: std > 0.03, Collapsing: std < 0.01
            scores["value_function_health"] = min(100, output_std * 2000)
        else:
            scores["value_function_health"] = 50
    else:
        scores["value_function_health"] = 50

    # --- Overall readiness ---
    known_scores = [v for v in scores.values() if v is not None]
    scores["overall"] = np.mean(known_scores) if known_scores else 0

    return scores


# ---------------------------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_diagnostic(diag, drift=None, scores=None, output_path="model_diagnostic.pdf",
                    model_name="Model", ref_name="Reference"):
    """Generate the full diagnostic visualization."""

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.06)

    title = f"Model Training Health Diagnostic — {model_name}"
    if diag["has_nan"]:
        title += "  ⚠ NaN DETECTED"
    if diag["has_inf"]:
        title += "  ⚠ Inf DETECTED"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)

    # Subtitle with architecture info
    arch_str = (f"Architecture: {diag['input_dim']}→{diag['embed_dim']}d, "
                f"{diag['num_layers']} layers, "
                f"{diag['total_params']:,} params "
                f"(enc {diag['encoder_params']:,} / dec {diag['decoder_params']:,} / "
                f"val {diag['value_params']:,})")
    fig.text(0.5, 0.925, arch_str, ha="center", fontsize=10, color="#555555")

    colors = {
        "good": "#27ae60",
        "ok": "#f39c12",
        "bad": "#e74c3c",
        "blue": "#2980b9",
        "purple": "#8e44ad",
        "teal": "#16a085",
        "gray": "#7f8c8d",
    }

    def score_color(score):
        if score >= 60:
            return colors["good"]
        elif score >= 30:
            return colors["ok"]
        return colors["bad"]

    def score_label(score):
        if score >= 60:
            return "Healthy"
        elif score >= 30:
            return "Developing"
        return "Undertrained"

    # === PANEL A: Overall Health Scores (top-left) ===
    ax_scores = fig.add_subplot(gs[0, 0])
    if scores:
        score_items = [
            ("Encoder specialization", scores.get("encoder_specialization", 0)),
            ("Decoder readiness", scores.get("decoder_readiness", 0)),
            ("LayerNorm adaptation", scores.get("layernorm_adaptation", 0)),
            ("Feature differentiation", scores.get("feature_differentiation", 0)),
            ("Value function health", scores.get("value_function_health", 50)),
        ]
        if scores.get("weight_drift") is not None:
            score_items.insert(0, ("Weight drift", scores["weight_drift"]))
        if scores.get("decoder_encoder_balance") is not None:
            score_items.append(("Dec/enc balance", scores["decoder_encoder_balance"]))

        labels = [s[0] for s in score_items]
        values = [s[1] for s in score_items]
        bar_colors = [score_color(v) for v in values]

        y_pos = np.arange(len(labels))
        bars = ax_scores.barh(y_pos, values, color=bar_colors, alpha=0.8,
                              height=0.6, edgecolor="white", linewidth=0.5)
        ax_scores.set_yticks(y_pos)
        ax_scores.set_yticklabels(labels, fontsize=9)
        ax_scores.set_xlim(0, 105)
        ax_scores.set_xlabel("Score (0-100)", fontsize=9)
        ax_scores.invert_yaxis()

        for bar, val in zip(bars, values):
            ax_scores.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                          f"{val:.0f}", va="center", fontsize=9, fontweight="bold")

        # Overall verdict
        overall = scores.get("overall", 0)
        verdict_color = score_color(overall)
        verdict_text = score_label(overall)
        ax_scores.text(52, len(labels) + 0.3,
                      f"Overall: {overall:.0f}/100 — {verdict_text}",
                      ha="center", fontsize=11, fontweight="bold",
                      color=verdict_color)
    ax_scores.set_title("A. Health scores", fontsize=11, fontweight="bold")
    ax_scores.grid(axis="x", alpha=0.2)

    # === PANEL B: Per-layer attention weight std (top-center-left) ===
    ax_layers = fig.add_subplot(gs[0, 1])
    n_layers = diag["num_layers"]
    x_layers = np.arange(n_layers)
    width = 0.18

    for idx, mat in enumerate(["W_q", "W_k", "W_v", "W_o"]):
        stds = [diag["layer_stats"][i].get(mat, {}).get("std", 0) for i in range(n_layers)]
        ax_layers.bar(x_layers + idx * width, stds, width,
                     label=mat, alpha=0.8)

    # Expected Xavier std line
    xavier = 1.0 / np.sqrt(diag["embed_dim"])
    ax_layers.axhline(xavier, color=colors["gray"], ls="--", lw=1, label=f"Xavier init ({xavier:.4f})")

    ax_layers.set_xticks(x_layers + 1.5 * width)
    ax_layers.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9)
    ax_layers.set_ylabel("Weight std", fontsize=9)
    ax_layers.legend(fontsize=7, ncol=3, loc="upper right")
    ax_layers.set_title("B. Attention weight std per layer", fontsize=11, fontweight="bold")
    ax_layers.grid(axis="y", alpha=0.2)

    # === PANEL C: Feature embedding analysis (top-center-right) ===
    ax_feat = fig.add_subplot(gs[0, 2])
    feat_stds = [fs["std"] for fs in diag["feature_stats"]]
    feat_labels = [f"F{i}" for i in range(diag["input_dim"])]
    bar_colors_feat = [colors["blue"]] * diag["input_dim"]

    if drift and "per_feature" in drift:
        # Color by drift magnitude
        max_drift = max(drift["per_feature"]) if drift["per_feature"] else 1
        for i, d in enumerate(drift["per_feature"]):
            if d < max_drift * 0.1:
                bar_colors_feat[i] = colors["bad"]
            elif d < max_drift * 0.5:
                bar_colors_feat[i] = colors["ok"]
            else:
                bar_colors_feat[i] = colors["good"]

    ax_feat.bar(feat_labels, feat_stds, color=bar_colors_feat, alpha=0.8,
                edgecolor="white", linewidth=0.5)
    ax_feat.set_ylabel("Embedding std", fontsize=9)
    ax_feat.set_xlabel("Input feature", fontsize=9)
    ax_feat.set_title("C. Per-feature embedding spread", fontsize=11, fontweight="bold")
    ax_feat.grid(axis="y", alpha=0.2)

    if drift and "per_feature" in drift:
        ax_feat2 = ax_feat.twinx()
        ax_feat2.plot(feat_labels, drift["per_feature"], "D-",
                     color=colors["purple"], markersize=5, linewidth=1.5,
                     label="Drift from ref")
        ax_feat2.set_ylabel("Drift from reference", fontsize=9, color=colors["purple"])
        ax_feat2.tick_params(axis="y", labelcolor=colors["purple"])
        ax_feat2.legend(fontsize=8, loc="upper left")

    # === PANEL D: Decoder vs Encoder comparison (top-right) ===
    ax_dec = fig.add_subplot(gs[0, 3])
    # Compare encoder and decoder attention weight stds
    enc_qkv_stds = []
    for i in range(n_layers):
        for mat in ["W_q", "W_k", "W_v"]:
            enc_qkv_stds.append(diag["layer_stats"][i].get(mat, {}).get("std", 0))

    dec_qkv_stds = [v.get("std", 0) for v in diag["decoder_stats"].values()]

    components = []
    comp_stds = []
    comp_colors = []

    for i in range(n_layers):
        for mat in ["W_q", "W_k", "W_v"]:
            components.append(f"Enc L{i}\n{mat}")
            comp_stds.append(diag["layer_stats"][i].get(mat, {}).get("std", 0))
            comp_colors.append(colors["blue"])

    for mat, stats in diag["decoder_stats"].items():
        components.append(f"Dec\n{mat}")
        comp_stds.append(stats.get("std", 0))
        comp_colors.append(colors["teal"])

    x_comp = np.arange(len(components))
    ax_dec.bar(x_comp, comp_stds, color=comp_colors, alpha=0.8,
               edgecolor="white", linewidth=0.5)
    ax_dec.axhline(xavier, color=colors["gray"], ls="--", lw=1, alpha=0.5)
    ax_dec.set_xticks(x_comp)
    ax_dec.set_xticklabels(components, fontsize=6, rotation=45, ha="right")
    ax_dec.set_ylabel("Weight std", fontsize=9)
    ax_dec.set_title("D. All attention weights: encoder vs decoder",
                     fontsize=11, fontweight="bold")
    ax_dec.grid(axis="y", alpha=0.2)

    # === PANEL E: Component drift (middle-left) — only if reference ===
    ax_drift = fig.add_subplot(gs[1, 0])
    if drift:
        comp_names = ["Embedding", "Encoder\nlayers", "Decoder", "Value\nhead"]
        comp_drifts = [drift["embedding"], drift["encoder_layers"],
                      drift["decoder"], drift["value_head"]]
        total = drift["total"]
        comp_pcts = [d / total * 100 if total > 0 else 0 for d in comp_drifts]

        drift_colors = [colors["teal"]] * len(comp_names)
        bars = ax_drift.bar(comp_names, comp_pcts, color=drift_colors, alpha=0.8,
                           edgecolor="white", linewidth=0.5)

        for bar, pct, val in zip(bars, comp_pcts, comp_drifts):
            ax_drift.text(bar.get_x() + bar.get_width()/2, pct + 1,
                         f"{pct:.1f}%\n({val:.1f})", ha="center", fontsize=8)

        ax_drift.set_ylabel("% of total drift", fontsize=9)
        ax_drift.set_title("E. Where did learning happen?", fontsize=11, fontweight="bold")
        ax_drift.grid(axis="y", alpha=0.2)
    else:
        ax_drift.text(0.5, 0.5, "No reference model\nprovided",
                     ha="center", va="center", fontsize=12, color=colors["gray"],
                     transform=ax_drift.transAxes)
        ax_drift.set_title("E. Weight drift (needs reference)", fontsize=11, fontweight="bold")

    # === PANEL F: Per-layer drift (middle-center-left) ===
    ax_ldrift = fig.add_subplot(gs[1, 1])
    if drift and "per_layer" in drift:
        layer_drifts = drift["per_layer"]
        layer_labels = [f"Layer {i}" for i in range(len(layer_drifts))]
        ldrift_colors = [colors["blue"] if d > np.mean(layer_drifts)
                        else colors["ok"] for d in layer_drifts]

        ax_ldrift.bar(layer_labels, layer_drifts, color=ldrift_colors, alpha=0.8,
                     edgecolor="white", linewidth=0.5)
        ax_ldrift.set_ylabel("L2 drift", fontsize=9)
        ax_ldrift.set_title("F. Learning per encoder layer", fontsize=11, fontweight="bold")
        ax_ldrift.grid(axis="y", alpha=0.2)

        # Add annotation
        max_layer = np.argmax(layer_drifts)
        ax_ldrift.annotate(f"Most active",
                          xy=(max_layer, layer_drifts[max_layer]),
                          xytext=(max_layer, layer_drifts[max_layer] * 1.15),
                          ha="center", fontsize=9, fontweight="bold",
                          color=colors["blue"])
    else:
        ax_ldrift.text(0.5, 0.5, "No reference model\nprovided",
                      ha="center", va="center", fontsize=12, color=colors["gray"],
                      transform=ax_ldrift.transAxes)
        ax_ldrift.set_title("F. Per-layer drift (needs reference)", fontsize=11, fontweight="bold")

    # === PANEL G: LayerNorm status (middle-center-right) ===
    ax_norm = fig.add_subplot(gs[1, 2])
    norm_data = {"weight_mean": [], "bias_mean": [], "weight_std": [], "bias_std": []}
    for i, ls in enumerate(diag["layer_stats"]):
        if "norm_w_mean" in ls:
            norm_data["weight_mean"].append(ls["norm_w_mean"])
            norm_data["bias_mean"].append(ls["norm_b_mean"])
            norm_data["weight_std"].append(ls["norm_w_std"])
            norm_data["bias_std"].append(ls["norm_b_std"])

    x_norm = np.arange(n_layers)
    w = 0.35
    ax_norm.bar(x_norm - w/2, norm_data["weight_mean"], w,
               label="γ (weight mean)", color=colors["blue"], alpha=0.8)
    ax_norm.bar(x_norm + w/2, norm_data["bias_mean"], w,
               label="β (bias mean)", color=colors["teal"], alpha=0.8)
    ax_norm.axhline(1.0, color=colors["gray"], ls="--", lw=1, alpha=0.5, label="Init γ=1")
    ax_norm.axhline(0.0, color=colors["gray"], ls=":", lw=1, alpha=0.5, label="Init β=0")
    ax_norm.set_xticks(x_norm)
    ax_norm.set_xticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9)
    ax_norm.set_ylabel("Parameter value", fontsize=9)
    ax_norm.legend(fontsize=7, ncol=2)
    ax_norm.set_title("G. LayerNorm parameters", fontsize=11, fontweight="bold")
    ax_norm.grid(axis="y", alpha=0.2)

    # === PANEL H: Std change heatmap (middle-right) ===
    ax_std = fig.add_subplot(gs[1, 3])
    if drift and "std_changes" in drift:
        # Group by component
        groups = OrderedDict()
        for key, pct in sorted(drift["std_changes"].items()):
            if "encoder.layers" in key:
                layer = key.split(".")[2]
                short = key.split(".")[-1].replace(".weight", "")
                group_key = f"Enc L{layer}"
            elif "decoder" in key:
                short = key.split(".")[-1].replace(".weight", "")
                group_key = "Decoder"
            elif "value" in key:
                short = key.split(".")[-1].replace(".weight", "")
                group_key = "Value"
            elif "embed" in key:
                short = "embed"
                group_key = "Embed"
            else:
                continue

            if group_key not in groups:
                groups[group_key] = {}
            # Simplify key name
            parts = key.replace("encoder.layers.", "").replace("decoder.", "").replace("value_head.", "")
            groups[group_key][parts] = pct

        # Flatten for display
        flat_names = []
        flat_values = []
        for group, items in list(groups.items())[:8]:  # limit display
            for name, val in list(items.items())[:4]:
                flat_names.append(f"{group}\n{name.split('.')[-1][:8]}")
                flat_values.append(val)

        if flat_values:
            bar_colors_std = [colors["good"] if v > 3 else
                             (colors["ok"] if v > 0 else colors["bad"])
                             for v in flat_values]
            x_std = np.arange(len(flat_names))
            ax_std.bar(x_std, flat_values, color=bar_colors_std, alpha=0.8,
                      edgecolor="white", linewidth=0.5)
            ax_std.axhline(0, color="black", lw=0.5)
            ax_std.set_xticks(x_std)
            ax_std.set_xticklabels(flat_names, fontsize=6, rotation=45, ha="right")
            ax_std.set_ylabel("Std change (%)", fontsize=9)
    ax_std.set_title("H. Weight std change from reference", fontsize=11, fontweight="bold")
    ax_std.grid(axis="y", alpha=0.2)

    # === PANEL I: Value head analysis (bottom-left) ===
    ax_val = fig.add_subplot(gs[2, 0])
    val_names = []
    val_stds = []
    val_means = []
    for key, stats in diag["value_stats"].items():
        short = key.replace("value_head.", "")
        val_names.append(short)
        val_stds.append(stats["std"])
        val_means.append(stats["mean"])

    if val_names:
        x_val = np.arange(len(val_names))
        ax_val.bar(x_val - 0.15, val_stds, 0.3, label="std", color=colors["blue"], alpha=0.8)
        ax_val.bar(x_val + 0.15, [abs(m) for m in val_means], 0.3,
                  label="|mean|", color=colors["teal"], alpha=0.8)
        ax_val.set_xticks(x_val)
        ax_val.set_xticklabels(val_names, fontsize=8)
        ax_val.legend(fontsize=8)
    ax_val.set_title("I. Value head parameters", fontsize=11, fontweight="bold")
    ax_val.grid(axis="y", alpha=0.2)

    # === PANEL J: Training readiness verdict (bottom, spanning 3 columns) ===
    ax_verdict = fig.add_subplot(gs[2, 1:])
    ax_verdict.axis("off")

    if scores:
        overall = scores.get("overall", 0)
        verdict_color = score_color(overall)
        verdict_text = score_label(overall)

        # Main verdict
        ax_verdict.text(0.5, 0.85, f"Overall Training Readiness: {overall:.0f}/100",
                       ha="center", va="center", fontsize=18, fontweight="bold",
                       color=verdict_color, transform=ax_verdict.transAxes)

        ax_verdict.text(0.5, 0.7, verdict_text.upper(),
                       ha="center", va="center", fontsize=14,
                       color=verdict_color, transform=ax_verdict.transAxes)

        # Detailed findings
        findings = []

        enc_spec = scores.get("encoder_specialization", 0)
        if enc_spec < 30:
            findings.append("• Encoder layers have NOT specialized — attention weights are near-identical across layers")
        elif enc_spec < 60:
            findings.append("• Encoder layers show SOME specialization — early signs of learning")
        else:
            findings.append("• Encoder layers show clear specialization ✓")

        dec_ready = scores.get("decoder_readiness", 0)
        if dec_ready < 20:
            findings.append("• Decoder is near initialization — likely producing uniform/saturated logits")
        elif dec_ready < 60:
            findings.append("• Decoder has begun adapting but may still lack discrimination")
        else:
            findings.append("• Decoder has moved well beyond initialization ✓")

        if drift:
            dec_pct = drift["decoder"] / drift["total"] * 100 if drift["total"] > 0 else 0
            if dec_pct < 15:
                findings.append(f"• Only {dec_pct:.0f}% of learning happened in decoder — policy output may be weak")
            enc_pct = drift["encoder_layers"] / drift["total"] * 100 if drift["total"] > 0 else 0
            max_layer = np.argmax(drift["per_layer"])
            findings.append(f"• Most encoder learning in layer {max_layer} ({drift['per_layer'][max_layer]:.1f} drift)")

        feat_diff = scores.get("feature_differentiation", 0)
        if feat_diff < 20:
            findings.append("• Input features are embedded nearly identically — model treats all features equally")
        elif feat_diff > 50:
            findings.append("• Input features show differentiated embeddings ✓")

        val_health = scores.get("value_function_health", 50)
        if val_health < 20:
            findings.append("• ⚠ Value function output may be collapsing (low weight std)")

        if overall < 30:
            findings.append("")
            findings.append("RECOMMENDATION: Train significantly longer. Current model is near initialization.")
        elif overall < 60:
            findings.append("")
            findings.append("RECOMMENDATION: Continue training. Model is developing but not yet converged.")

        finding_text = "\n".join(findings)
        ax_verdict.text(0.05, 0.5, finding_text, ha="left", va="center",
                       fontsize=10, transform=ax_verdict.transAxes,
                       family="monospace", linespacing=1.6)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nDiagnostic saved to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Model Training Health Diagnostic"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("--reference", "-r", default=None,
                        help="Reference checkpoint for drift comparison "
                             "(e.g., initialization or earlier checkpoint)")
    parser.add_argument("--output", "-o", default="results/figures/model_diagnostic.pdf",
                        help="Output path for diagnostic plot")
    parser.add_argument("--name", default=None,
                        help="Model name for title")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_name = args.name or os.path.basename(args.checkpoint)

    # Analyze
    print("Analyzing model...")
    diag = analyze_checkpoint(state_dict)

    # Load reference if provided
    drift = None
    if args.reference:
        print(f"Loading reference: {args.reference}")
        ref_dict = torch.load(args.reference, map_location="cpu", weights_only=False)
        print("Computing drift...")
        drift = compute_drift(state_dict, ref_dict)

    # Compute scores
    scores = compute_health_scores(diag, drift)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Architecture: {diag['input_dim']}→{diag['embed_dim']}d, "
          f"{diag['num_layers']} layers, {diag['total_params']:,} params")
    print(f"NaN: {diag['has_nan']} | Inf: {diag['has_inf']}")
    print()
    for name, score in scores.items():
        if score is not None:
            indicator = "✓" if score >= 60 else ("~" if score >= 30 else "✗")
            print(f"  {indicator} {name}: {score:.1f}/100")
    print()
    overall = scores.get("overall", 0)
    label = "Healthy" if overall >= 60 else ("Developing" if overall >= 30 else "Undertrained")
    print(f"  Overall: {overall:.0f}/100 — {label}")

    # Plot
    print("\nGenerating diagnostic plot...")
    plot_diagnostic(diag, drift, scores, args.output, model_name)


if __name__ == "__main__":
    main()
