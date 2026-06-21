import os
import sys
import csv
import time
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Add project root to path
# ------------------------------------------------------------
framework_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(framework_path))

# ------------------------------------------------------------
# Import your RL4EVRP framework
# ------------------------------------------------------------
import rl4evrp as rl
from rl4evrp.utils import train_agent, run_episode


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ALL_FEATURES = [
    "x",                     # 0
    "y",                     # 1
    "demand_normalized",     # 2
    "is_charger",            # 3
    "is_depot",              # 4
    "cargo_capacity_norm",   # 5
    "battery_capacity_norm", # 6
    "tw_ready_norm",         # 7
    "tw_due_norm",           # 8
    "service_time_norm",     # 9
]

ABLATION_CONFIGS = OrderedDict([
    ("Full (all 10)",         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("No time windows",       [0, 1, 2, 3, 4, 5, 6]),
    ("No battery state",      [0, 1, 2, 3, 4, 5, 7, 8, 9]),
    ("No cargo state",        [0, 1, 2, 3, 4, 6, 7, 8, 9]),
    ("No node type flags",    [0, 1, 2, 5, 6, 7, 8, 9]),
    ("No demand",             [0, 1, 3, 4, 5, 6, 7, 8, 9]),
    ("Spatial + demand only", [0, 1, 2]),
])


# ============================================================
# HELPERS
# ============================================================

def make_feature_mask(keep_indices, total_features=10):
    """
    Create a binary feature mask of length `total_features`.
    1 = keep feature, 0 = mask to zero.
    """
    mask = np.zeros(total_features, dtype=np.float32)
    for i in keep_indices:
        mask[i] = 1.0
    return mask


def attach_feature_mask(instances, feature_mask):
    """
    Return copies of instances with the feature mask attached.
    The environment should read inst['feature_mask'] and apply it.
    """
    out = []
    for inst in instances:
        new_inst = dict(inst)
        new_inst["feature_mask"] = np.array(feature_mask, dtype=np.float32)
        out.append(new_inst)
    return out


def format_config_dir_name(config_name):
    """
    Safe directory name for saving checkpoints/results.
    """
    name = config_name.replace(" ", "_")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("+", "plus")
    name = name.replace("/", "_")
    return name


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_with_mask(
    model_builder,
    train_instances,
    eval_instances,
    feature_mask,
    config_name,
    n_episodes,
    device,
    save_dir,
    seed=42,
):
    """
    Train one model under one ablation mask.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fresh model for each ablation
    model = model_builder.complete_model()

    masked_train_instances = attach_feature_mask(train_instances, feature_mask)
    masked_eval_instances = attach_feature_mask(eval_instances, feature_mask)

    active_features = [ALL_FEATURES[i] for i in np.where(feature_mask == 1)[0]]
    masked_features = [ALL_FEATURES[i] for i in np.where(feature_mask == 0)[0]]

    print(f"\n{'=' * 70}")
    print(f"Training config: {config_name}")
    print(f"Active features: {active_features}")
    print(f"Masked features: {masked_features}")
    print(f"{'=' * 70}")

    config_save_dir = Path(save_dir) / f"ablation_{format_config_dir_name(config_name)}"
    config_save_dir.mkdir(parents=True, exist_ok=True)

    training_results = train_agent(
        model,
        masked_train_instances,
        n_episodes=n_episodes,
        device=device,
        save_dir=config_save_dir,
        eval_instances=masked_eval_instances,
        save_interval=n_episodes,  # save only final checkpoint
    )

    return model, training_results


def evaluate_with_mask(model, eval_instances, feature_mask, device='cpu', n_eval=100):
    """
    Evaluate one trained model under one ablation mask.
    """
    distances = []
    served_list = []
    feasible_list = []
    charger_list = []

    masked_eval_instances = attach_feature_mask(eval_instances, feature_mask)

    for i in range(min(n_eval, len(masked_eval_instances))):
        inst = dict(masked_eval_instances[i], reward_mode='distance')

        with torch.no_grad():
            _, _, dist, info, _, _, env = run_episode(
                model,
                inst,
                greedy=True
            )

        distances.append(float(dist))

        n_cust = int((np.array(inst['node_types']) == 1).sum())
        served = int(info.get('n_customers_served', 0))
        feasible = 1 if served == n_cust else 0
        charger_visits = int(info.get('charger_visits', 0))

        served_list.append(served)
        feasible_list.append(feasible)
        charger_list.append(charger_visits)

    return {
        'distance_mean': float(np.mean(distances)),
        'distance_std': float(np.std(distances)),
        'served_mean': float(np.mean(served_list)),
        'feasible_rate': float(np.mean(feasible_list)),
        'charger_mean': float(np.mean(charger_list)),
    }


# ============================================================
# MAIN STUDY
# ============================================================

def run_ablation_study(
    model_builder,
    framework,
    train_instances,
    eval_instances,
    n_episodes=50000,
    n_eval=100,
    device='cpu',
    output_dir='ablation_results',
    seed=42,
):
    """
    Run all ablation configurations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = OrderedDict()

    for config_name, keep_indices in ABLATION_CONFIGS.items():
        feature_mask = make_feature_mask(keep_indices)
        n_active = int(feature_mask.sum())

        print(f"\n{'#' * 70}")
        print(f"# ABLATION: {config_name} ({n_active}/10 features)")
        print(f"{'#' * 70}")

        t0 = time.time()

        # Train
        model, _ = train_with_mask(
            model_builder=model_builder,
            train_instances=train_instances,
            eval_instances=eval_instances,
            feature_mask=feature_mask,
            config_name=config_name,
            n_episodes=n_episodes,
            device=device,
            save_dir=output_dir,
            seed=seed,
        )

        # Evaluate
        eval_results = evaluate_with_mask(
            model=model,
            eval_instances=eval_instances,
            feature_mask=feature_mask,
            device=device,
            n_eval=n_eval,
        )

        elapsed = time.time() - t0

        eval_results['config'] = config_name
        eval_results['n_features'] = n_active
        eval_results['kept_features'] = str([ALL_FEATURES[i] for i in keep_indices])
        eval_results['train_time_min'] = elapsed / 60.0

        results[config_name] = eval_results

        print(f"Distance: {eval_results['distance_mean']:.3f} ± {eval_results['distance_std']:.3f}")
        print(f"Served:   {eval_results['served_mean']:.2f}")
        print(f"Feasible: {eval_results['feasible_rate']:.0%}")
        print(f"Chargers: {eval_results['charger_mean']:.2f}")
        print(f"Time:     {elapsed / 60:.1f} min")

    # Save CSV
    csv_path = output_dir / 'state_ablation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(list(results.values())[0].keys()))
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

    print(f"\nResults saved to {csv_path}")

    # Save figure
    fig_path = output_dir / 'state_ablation_figure.pdf'
    plot_ablation_results(results, fig_path)

    # Save LaTeX table
    tex_path = output_dir / 'state_ablation_table.tex'
    generate_latex_table(results, tex_path)

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_ablation_results(results, output_path):
    """
    Create paper-ready figure with:
    (a) mean route distance
    (b) served customers + feasibility
    """
    configs = list(results.keys())
    distances = [results[c]['distance_mean'] for c in configs]
    dist_stds = [results[c]['distance_std'] for c in configs]
    feasibility = [results[c]['feasible_rate'] * 100 for c in configs]
    served = [results[c]['served_mean'] for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(configs))

    colors = []
    baseline = distances[0]
    for i, d in enumerate(distances):
        if i == 0:
            colors.append('#27ae60')  # full model = green
        elif d > baseline * 1.15:
            colors.append('#c0392b')  # bad degradation = red
        elif d > baseline * 1.05:
            colors.append('#f39c12')  # moderate degradation = orange
        else:
            colors.append('#3498db')  # close to baseline = blue

    # Panel A
    bars1 = ax1.bar(
        x, distances,
        color=colors,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5,
        width=0.65
    )
    ax1.errorbar(
        x, distances,
        yerr=dist_stds,
        fmt='none',
        ecolor='#2c3e50',
        capsize=3,
        capthick=1,
        linewidth=1
    )
    ax1.axhline(baseline, color='#27ae60', ls='--', lw=1, alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Average distance', fontsize=11)
    ax1.set_title('(a) Route distance by feature configuration',
                  fontsize=12, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.15)

    for bar, val in zip(bars1, distances):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.1,
            f'{val:.2f}',
            ha='center',
            fontsize=8,
            fontweight='bold'
        )

    # Panel B
    width = 0.35
    ax2.bar(
        x - width/2, served, width,
        color='#3498db',
        alpha=0.85,
        label='Customers served',
        edgecolor='white'
    )
    ax2.bar(
        x + width/2, feasibility, width,
        color='#27ae60',
        alpha=0.85,
        label='Feasibility (%)',
        edgecolor='white'
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=35, ha='right', fontsize=9)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('(b) Service completion and feasibility',
                  fontsize=12, fontweight='bold', loc='left')
    ax2.legend(fontsize=9, loc='lower left', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved to {output_path}")


# ============================================================
# LATEX TABLE
# ============================================================

def generate_latex_table(results, output_path):
    """
    Generate a LaTeX table summarizing the ablation results.
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Feat. & Dist. & Served & Feas. & Chg. \\")
    lines.append(r"\midrule")

    configs = list(results.keys())
    baseline_dist = results[configs[0]]['distance_mean']

    for i, config in enumerate(configs):
        r = results[config]
        if i == 0:
            lines.append(
                f"\\textbf{{{config}}} & "
                f"\\textbf{{{r['n_features']}}} & "
                f"\\textbf{{{r['distance_mean']:.2f}}} & "
                f"\\textbf{{{r['served_mean']:.1f}}} & "
                f"\\textbf{{{r['feasible_rate']:.0%}}} & "
                f"\\textbf{{{r['charger_mean']:.1f}}} \\\\"
            )
        else:
            delta = (r['distance_mean'] - baseline_dist) / baseline_dist * 100
            delta_str = f"({delta:+.1f}\\%)"
            lines.append(
                f"{config} & "
                f"{r['n_features']} & "
                f"{r['distance_mean']:.2f} {delta_str} & "
                f"{r['served_mean']:.1f} & "
                f"{r['feasible_rate']:.0%} & "
                f"{r['charger_mean']:.1f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{State feature ablation for RL4EVRP. "
        r"Each variant masks selected input channels to zero while keeping "
        r"the model architecture and training settings fixed.}"
    )
    lines.append(r"\label{tab:state_ablation}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to {output_path}")


# ============================================================
# CLI / SCRIPT ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="State feature ablation for RL4EVRP")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Training episodes per ablation config")
    parser.add_argument("--eval-instances", type=int, default=100,
                        help="Number of evaluation instances")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Override number of training instances")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                        help="Directory to save outputs")
    args = parser.parse_args()

    print("✓ RL4EVRP framework imported successfully")

    # Initialize framework
    framework = rl.RL4EVRP()

    # Read configs
    problem_config = framework.read_yaml('problem')
    model_config = framework.read_yaml('model')
    env_config = framework.read_yaml('env')

    print("Problem Configuration:")
    framework.config.print_config()

    # Build model
    model_builder = framework.build()
    model = model_builder.complete_model()

    print(f"Model device: {model.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate instances
    instance_cfg = problem_config['instance_generation']
    n_train = args.train_size if args.train_size is not None else instance_cfg.get(
        'train_size',
        instance_cfg.get('n_train_instances', 50000)
    )
    n_eval = args.eval_instances
    seed_offset = instance_cfg.get('seed_offset', 0)

    train_instances = [
        framework.generate_instance(seed=seed_offset + i)
        for i in range(n_train)
    ]

    eval_instances = [
        framework.generate_instance(seed=1000 + seed_offset + i)
        for i in range(n_eval)
    ]

    print(f"✓ Generated {len(train_instances)} training instances")
    print(f"✓ Generated {len(eval_instances)} evaluation instances")

    demo_inst = train_instances[0]
    print("\nExample instance:")
    print(f"  Nodes: {demo_inst['n_nodes']}")
    print(f"  Chargers: {(demo_inst['node_types'] == 2).sum()}")
    print(f"  Customers: {(demo_inst['node_types'] == 1).sum()}")

    # Run study
    run_ablation_study(
        model_builder=model_builder,
        framework=framework,
        train_instances=train_instances,
        eval_instances=eval_instances,
        n_episodes=args.episodes,
        n_eval=n_eval,
        device=str(framework.device),
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()