import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add framework to path
framework_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(framework_path))

import rl4evrp as rl
from rl4evrp.utils import run_episode


# ============================================================================
# GENERATE EVRP INSTANCE
# ============================================================================

def generate_evrp_instance(
    n_customers=20,
    n_chargers=3,
    seed=42,
    cargo_cap=30.0,
    battery_cap=100.0,
):
    rng = np.random.RandomState(seed)

    depot = rng.uniform(0.3, 0.7, size=2).astype(np.float32)
    customers = rng.uniform(0.05, 0.95, size=(n_customers, 2)).astype(np.float32)
    chargers = rng.uniform(0.1, 0.9, size=(n_chargers, 2)).astype(np.float32)
    demands = rng.randint(1, 10, size=n_customers).astype(np.float32)

    coords = np.vstack([depot.reshape(1, 2), customers, chargers]).astype(np.float32)
    node_types = np.array(
        [0] + [1] * n_customers + [2] * n_chargers,
        dtype=np.int64
    )
    all_demands = np.concatenate([
        [0.0],
        demands,
        np.zeros(n_chargers, dtype=np.float32)
    ]).astype(np.float32)

    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float32)

    inst = {
        "coords": coords,
        "locs": coords,
        "locations": coords,

        "node_types": node_types,

        "demands": all_demands,
        "demand": all_demands,

        "cargo_cap": float(cargo_cap),
        "cargo_capacity": float(cargo_cap),

        "battery_cap": float(battery_cap),
        "battery_capacity": float(battery_cap),

        "distance_matrix": D,
        "dist_matrix": D,
        "D": D,

        "n_nodes": int(len(coords)),
        "n_customers": int(n_customers),
        "n_chargers": int(n_chargers),
        "depot_idx": 0,
    }

    return inst


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_path):
    framework = rl.RL4EVRP()
    framework.read_yaml("problem")
    framework.read_yaml("model")
    framework.read_yaml("env")

    model_builder = framework.build()
    model = model_builder.complete_model()

    state_dict = torch.load(model_path, map_location=framework.device)
    model.load_state_dict(state_dict)
    model.to(framework.device)
    model.eval()

    return framework, model


# ============================================================================
# PLOT EXACT STYLE
# ============================================================================

def plot_route_exact_style(inst, route, reward, dist, info, env, output_path, title="EVRP Route"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    coords = inst['coords']
    ntypes = inst['node_types']

    # ── Route map ─────────────────────────────────────────────────────────────
    ax = axes[0]
    for i in range(len(route) - 1):
        n1, n2 = route[i], route[i + 1]
        col_e = '#2ECC71' if ntypes[n2] == 2 else ('red' if n2 == 0 else '#3498DB')
        ax.annotate('', xy=coords[n2], xytext=coords[n1],
                    arrowprops=dict(arrowstyle='->', color=col_e, lw=2, alpha=0.7))

    for n in range(env.n):
        x, y = coords[n]
        if n == 0:
            ax.plot(x, y, '*', color='red', ms=22, zorder=6)
            ax.annotate('Depot', (x, y), textcoords='offset points',
                        xytext=(8, 8), fontsize=9, color='red', fontweight='bold')
        elif ntypes[n] == 1:
            sz = 80 + inst['demands'][n] * 20
            ax.scatter(x, y, s=sz, color='#3498DB', zorder=5,
                       edgecolors='white', lw=1.5)
            ax.annotate(f'C{n}\n({inst["demands"][n]:.0f})', (x, y),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=7, color='#2980B9')
        else:
            ax.plot(x, y, 's', color='#2ECC71', ms=13, zorder=5)
            ax.annotate(f'CH{n}', (x, y), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, color='#27AE60')

    for i, n in enumerate(route[:25]):
        ax.text(coords[n, 0] + 0.02, coords[n, 1] + 0.03,
                str(i), fontsize=7, color='gray', ha='center')

    customers = np.where(ntypes == 1)[0]
    n_served = sum(1 for n in route if ntypes[n] == 1)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Route Map | Total dist={env.total_d:.3f}\n'
                 f'Served={n_served}/{len(customers)}  Chargers={env.charger_visits}')
    ax.legend(handles=[
        mpatches.Patch(color='red',     label='Depot'),
        mpatches.Patch(color='#3498DB', label='Customer (size=demand)'),
        mpatches.Patch(color='#2ECC71', label='Charger'),
    ], fontsize=8)
    ax.grid(alpha=0.2)

    # ── Battery trace / extra info ───────────────────────────────────────────
    ax2 = axes[1]

    if hasattr(env, "battery_log") and len(env.battery_log) > 0:
        ax2.plot(env.battery_log, lw=2)
        ax2.set_title("Battery level over route")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Battery")
        ax2.grid(alpha=0.2)
    else:
        txt = [
            f"Reward: {reward:.4f}",
            f"Distance: {dist:.4f}",
            f"Battery violations: {info.get('batt_violations', 'N/A')}",
            f"Cargo violations: {info.get('cargo_violations', 'N/A')}",
            f"Charger visits: {info.get('charger_visits', 'N/A')}",
            f"Route length: {len(route)}",
        ]
        ax2.axis("off")
        ax2.text(0.05, 0.95, "\n".join(txt), va="top", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--output", default="evrp_route.png", help="Plot output file")
    parser.add_argument("--seed", type=int, default=42, help="Instance seed")
    parser.add_argument("--n-customers", type=int, default=20)
    parser.add_argument("--n-chargers", type=int, default=3)
    parser.add_argument("--cargo-cap", type=float, default=30.0)
    parser.add_argument("--battery-cap", type=float, default=100.0)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use sampling instead of greedy")
    args = parser.parse_args()

    print("Loading model...")
    framework, model = load_model(args.model)
    print(f"✓ Model loaded on device: {framework.device}")

    print(f"Generating EVRP instance with seed={args.seed}")
    inst = generate_evrp_instance(
        n_customers=args.n_customers,
        n_chargers=args.n_chargers,
        seed=args.seed,
        cargo_cap=args.cargo_cap,
        battery_cap=args.battery_cap,
    )

    print("Running policy...")
    reward, route, dist, info, _, _, env = run_episode(
        model,
        inst,
        device=str(framework.device),
        greedy=True
    )

    title = f"EVRP Test | seed={args.seed} | dist={dist:.3f} | reward={reward:.3f}"
    plot_route_exact_style(inst, route, reward, dist, info, env, args.output, title=title)

    print("\n=== RESULT ===")
    print(f"Seed: {args.seed}")
    print(f"Reward: {reward:.6f}")
    print(f"Distance: {dist:.6f}")
    print(f"Route: {route}")
    print(f"Battery violations: {info.get('batt_violations', 'N/A')}")
    print(f"Cargo violations: {info.get('cargo_violations', 'N/A')}")
    print(f"Charger visits: {info.get('charger_visits', 'N/A')}")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()