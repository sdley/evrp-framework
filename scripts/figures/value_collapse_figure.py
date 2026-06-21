"""
Value Head Collapse Figure — Paper Ready

Plots weight std trends across checkpoints showing:
- Value head collapsing (red, declining)
- Encoder growing (green, increasing)  
- Decoder stagnating (blue, flat)

Usage:
    python value_collapse_figure.py \
        --checkpoints cp1.pt cp2.pt cp3.pt cp4.pt \
        --episodes 800 18000 26800 49040 \
        --output value_head_collapse.pdf
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_figure(checkpoint_paths, episode_counts, output_path):
    val_stds = []
    dec_stds = []
    enc0_stds = []

    for path in checkpoint_paths:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        val_stds.append(sd['value_head.2.weight'].std().item())
        dec_stds.append(sd['decoder.cross_attn.W_q.weight'].std().item())
        enc0_stds.append(sd['encoder.layers.0.attn.W_q.weight'].std().item())

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Plot lines
    ax.plot(episode_counts, val_stds, 'o-', color='#c0392b', linewidth=2.5,
            markersize=9, label='Value head output (critic)', zorder=5)
    ax.plot(episode_counts, dec_stds, 's-', color='#2980b9', linewidth=2.5,
            markersize=8, label='Decoder cross-attention', zorder=4)
    ax.plot(episode_counts, enc0_stds, '^-', color='#27ae60', linewidth=2.5,
            markersize=8, label='Encoder layer 0 attention', zorder=4)

    # Value annotations
    for ep, vs in zip(episode_counts, val_stds):
        ax.annotate(f'{vs:.3f}', (ep, vs), textcoords='offset points',
                    xytext=(0, -18), ha='center', fontsize=9, color='#c0392b',
                    fontweight='bold')

    # Encoder annotations
    for ep, es in zip(episode_counts, enc0_stds):
        ax.annotate(f'{es:.3f}', (ep, es), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=9, color='#27ae60',
                    fontweight='bold')

    # Reference line
    xavier = 1.0 / np.sqrt(128)
    ax.axhline(xavier, color='gray', ls='--', lw=1, alpha=0.5,
               label=f'Xavier init ({xavier:.4f})')

    # Collapse arrow
    ax.annotate('Value collapse', 
                xy=(episode_counts[-1], val_stds[-1]),
                xytext=(episode_counts[-1] - 12000, val_stds[-1] - 0.008),
                fontsize=11, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

    # Growth arrow
    ax.annotate('Encoder growing',
                xy=(episode_counts[-1], enc0_stds[-1]),
                xytext=(episode_counts[-1] - 12000, enc0_stds[-1] + 0.008),
                fontsize=11, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    # Stagnation arrow
    ax.annotate('Decoder stagnant',
                xy=(episode_counts[-1], dec_stds[-1]),
                xytext=(episode_counts[-1] - 15000, dec_stds[-1] + 0.012),
                fontsize=11, color='#2980b9', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

    ax.set_xlabel('Training episodes', fontsize=13)
    ax.set_ylabel('Weight standard deviation', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.15)
    ax.set_xlim(0, max(episode_counts) * 1.08)
    ax.set_ylim(0.030, 0.080)

    # Legend outside the plot area, below
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # make room for legend below

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths in order")
    parser.add_argument("--episodes", nargs="+", type=int, required=True,
                        help="Episode counts matching checkpoints")
    parser.add_argument("--output", default="results/figures/value_head_collapse.pdf")
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.episodes), \
        "Must provide same number of checkpoints and episode counts"

    make_figure(args.checkpoints, args.episodes, args.output)
