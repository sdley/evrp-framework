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
# Import RL4EVRP framework
# ------------------------------------------------------------
import rl4evrp as rl
from rl4evrp.utils import train_agent, run_episode


# ============================================================
# STATE VARIANTS
# Keep feature_dim fixed to 10 so the model architecture
# stays unchanged across variants.
# ============================================================

STATE_VARIANTS = OrderedDict([
    (
        "Original full",
        [
            "x",
            "y",
            "demand_normalized",
            "is_charger",
            "is_depot",
            "cargo_capacity_norm",
            "battery_capacity_norm",
            "tw_ready_norm",
            "tw_due_norm",
            "service_time_norm",
        ],
    ),
    (
        "Static graph",
        [
            "x",
            "y",
            "demand_normalized",
            "is_charger",
            "is_depot",
            "zeros",
            "zeros",
            "zeros",
            "zeros",
            "zeros",
        ],
    ),
    (
        "Spatial+time",
        [
            "x",
            "y",
            "demand_normalized",
            "is_charger",
            "is_depot",
            "tw_ready_norm",
            "tw_due_norm",
            "service_time_norm",
            "zeros",
            "zeros",
        ],
    ),
    (
        "Resource-aware",
        [
            "x",
            "y",
            "demand_normalized",
            "is_charger",
            "is_depot",
            "cargo_capacity_norm",
            "battery_capacity_norm",
            "demand_over_cargo",
            "zeros",
            "zeros",
        ],
    ),
    (
        "Constraint-aware",
        [
            "x",
            "y",
            "demand_normalized",
            "is_charger",
            "is_depot",
            "cargo_capacity_norm",
            "battery_capacity_norm",
            "tw_window_width_norm",
            "tw_due_norm",
            "service_time_norm",
        ],
    ),
])


# ============================================================
# HELPERS
# ============================================================

def attach_state_variant(instances, variant_name, feature_names):
    """
    Attach state-variant metadata to instances.
    The environment must read:
      - inst['state_variant']
      - inst['state_feature_names']
    """
    out = []
    for inst in instances:
        new_inst = dict(inst)
        new_inst["state_variant"] = variant_name
        new_inst["state_feature_names"] = list(feature_names)
        out.append(new_inst)
    return out


def format_config_dir_name(name):
    name = name.replace(" ", "_")
    name = name.replace("+", "plus")
    name = name.replace("/", "_")
    name = name.replace("(", "").replace(")", "")
    return name


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_state_variant(
    model_builder,
    train_instances,
    eval_instances,
    variant_name,
    feature_names,
    n_episodes,
    device,
    save_dir,
    seed=42,
):
    set_seed(seed)

    model = model_builder.complete_model()

    train_variant = attach_state_variant(train_instances, variant_name, feature_names)
    eval_variant = attach_state_variant(eval_instances, variant_name, feature_names)

    print(f"\n{'=' * 72}")
    print(f"Training state variant: {variant_name}")
    print(f"Features: {feature_names}")
    print(f"{'=' * 72}")

    variant_dir = Path(save_dir) / f"state_{format_config_dir_name(variant_name)}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    training_results = train_agent(
        model,
        train_variant,
        n_episodes=n_episodes,
        device=device,
        save_dir=variant_dir,
        eval_instances=eval_variant,
        save_interval=n_episodes,
    )

    return model, training_results


def evaluate_with_state_variant(
    model,
    eval_instances,
    variant_name,
    feature_names,
    n_eval=100,
):
    distances = []
    served_list = []
    feasible_list = []
    charger_list = []

    eval_variant = attach_state_variant(eval_instances, variant_name, feature_names)

    for i in range(min(n_eval, len(eval_variant))):
        inst = dict(eval_variant[i], reward_mode="distance")

        with torch.no_grad():
            _, _, dist, info, _, _, env = run_episode(
                model,
                inst,
                greedy=True
            )

        distances.append(float(dist))

        n_cust = int((np.array(inst["node_types"]) == 1).sum())
        served = int(info.get("n_customers_served", 0))
        feasible = 1 if served == n_cust else 0
        charger_visits = int(info.get("charger_visits", 0))

        served_list.append(served)
        feasible_list.append(feasible)
        charger_list.append(charger_visits)

    return {
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "served_mean": float(np.mean(served_list)),
        "feasible_rate": float(np.mean(feasible_list)),
        "charger_mean": float(np.mean(charger_list)),
    }


# ============================================================
# MAIN STUDY
# ============================================================

def run_state_study(
    model_builder,
    train_instances,
    eval_instances,
    n_episodes=50000,
    n_eval=100,
    device="cpu",
    output_dir="state_formulation_results",
    seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = OrderedDict()

    for variant_name, feature_names in STATE_VARIANTS.items():
        print(f"\n{'#' * 72}")
        print(f"# STATE VARIANT: {variant_name}")
        print(f"{'#' * 72}")

        t0 = time.time()

        model, _ = train_with_state_variant(
            model_builder=model_builder,
            train_instances=train_instances,
            eval_instances=eval_instances,
            variant_name=variant_name,
            feature_names=feature_names,
            n_episodes=n_episodes,
            device=device,
            save_dir=output_dir,
            seed=seed,
        )

        eval_results = evaluate_with_state_variant(
            model=model,
            eval_instances=eval_instances,
            variant_name=variant_name,
            feature_names=feature_names,
            n_eval=n_eval,
        )

        elapsed = time.time() - t0

        eval_results["config"] = variant_name
        eval_results["n_features"] = len(feature_names)
        eval_results["feature_names"] = str(feature_names)
        eval_results["train_time_min"] = elapsed / 60.0

        results[variant_name] = eval_results

        print(f"Distance: {eval_results['distance_mean']:.3f} ± {eval_results['distance_std']:.3f}")
        print(f"Served:   {eval_results['served_mean']:.2f}")
        print(f"Feasible: {eval_results['feasible_rate']:.0%}")
        print(f"Chargers: {eval_results['charger_mean']:.2f}")
        print(f"Time:     {elapsed / 60:.1f} min")

    csv_path = output_dir / "state_formulation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(list(results.values())[0].keys()))
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

    print(f"\nResults saved to {csv_path}")

    fig_path = output_dir / "state_formulation_figure.pdf"
    plot_state_results(results, fig_path)

    tex_path = output_dir / "state_formulation_table.tex"
    generate_latex_table(results, tex_path)

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_state_results(results, output_path):
    configs = list(results.keys())
    distances = [results[c]["distance_mean"] for c in configs]
    dist_stds = [results[c]["distance_std"] for c in configs]
    feasibility = [results[c]["feasible_rate"] * 100 for c in configs]
    served = [results[c]["served_mean"] for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(configs))

    baseline = distances[0]
    colors = []
    for i, d in enumerate(distances):
        if i == 0:
            colors.append("#27ae60")
        elif d > baseline * 1.15:
            colors.append("#c0392b")
        elif d > baseline * 1.05:
            colors.append("#f39c12")
        else:
            colors.append("#3498db")

    bars1 = ax1.bar(
        x, distances,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.65
    )
    ax1.errorbar(
        x, distances,
        yerr=dist_stds,
        fmt="none",
        ecolor="#2c3e50",
        capsize=3,
        capthick=1,
        linewidth=1
    )
    ax1.axhline(baseline, color="#27ae60", ls="--", lw=1, alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Average distance", fontsize=11)
    ax1.set_title("(a) Route distance by state formulation",
                  fontsize=12, fontweight="bold", loc="left")
    ax1.grid(axis="y", alpha=0.15)

    for bar, val in zip(bars1, distances):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.1,
            f"{val:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

    width = 0.35
    ax2.bar(
        x - width/2, served, width,
        alpha=0.85,
        label="Customers served",
        edgecolor="white"
    )
    ax2.bar(
        x + width/2, feasibility, width,
        alpha=0.85,
        label="Feasibility (%)",
        edgecolor="white"
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("(b) Service completion and feasibility",
                  fontsize=12, fontweight="bold", loc="left")
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved to {output_path}")


# ============================================================
# LATEX TABLE
# ============================================================

def generate_latex_table(results, output_path):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"State formulation & Dist. & Served & Feas. & Chg. \\")
    lines.append(r"\midrule")

    configs = list(results.keys())
    baseline_dist = results[configs[0]]["distance_mean"]

    for i, config in enumerate(configs):
        r = results[config]
        if i == 0:
            lines.append(
                f"\\textbf{{{config}}} & "
                f"\\textbf{{{r['distance_mean']:.2f}}} & "
                f"\\textbf{{{r['served_mean']:.1f}}} & "
                f"\\textbf{{{r['feasible_rate']:.0%}}} & "
                f"\\textbf{{{r['charger_mean']:.1f}}} \\\\"
            )
        else:
            delta = (r["distance_mean"] - baseline_dist) / baseline_dist * 100
            delta_str = f"({delta:+.1f}\\%)"
            lines.append(
                f"{config} & "
                f"{r['distance_mean']:.2f} {delta_str} & "
                f"{r['served_mean']:.1f} & "
                f"{r['feasible_rate']:.0%} & "
                f"{r['charger_mean']:.1f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Comparison of alternative state formulations for RL4EVRP. "
        r"All experiments use the same model architecture, reward, and training setup; "
        r"only the configured state definition differs.}"
    )
    lines.append(r"\label{tab:state_formulation}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to {output_path}")


# ============================================================
# CLI / SCRIPT ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="State formulation study for RL4EVRP")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Training episodes per state variant")
    parser.add_argument("--eval-instances", type=int, default=100,
                        help="Number of evaluation instances")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Override number of training instances")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/studies/state_formulation",
                        help="Directory to save outputs")
    args = parser.parse_args()

    print("✓ RL4EVRP framework imported successfully")

    framework = rl.RL4EVRP()

    problem_config = framework.read_yaml("problem")
    model_config = framework.read_yaml("model")
    env_config = framework.read_yaml("env")

    print("Problem Configuration:")
    framework.config.print_config()

    model_builder = framework.build()
    model = model_builder.complete_model()

    print(f"Model device: {model.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    instance_cfg = problem_config["instance_generation"]
    n_train = args.train_size if args.train_size is not None else instance_cfg.get(
        "train_size",
        instance_cfg.get("n_train_instances", 50000)
    )
    n_eval = args.eval_instances
    seed_offset = instance_cfg.get("seed_offset", 0)

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

    run_state_study(
        model_builder=model_builder,
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