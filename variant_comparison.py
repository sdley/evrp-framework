import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add framework to path
framework_path = Path.cwd()
sys.path.insert(0, str(framework_path))

import rl4evrp as rl
from rl4evrp.utils import run_episode


# ============================================================================
# SHARED INSTANCE GENERATION
# ============================================================================

def generate_shared_instance(
    n_customers=15,
    n_chargers=3,
    seed=42,
    cargo_capacity=30.0,
    battery_capacity=100.0,
):
    rng = np.random.RandomState(seed)

    depot = rng.uniform(0.3, 0.7, size=2).astype(np.float32)
    customers = rng.uniform(0.05, 0.95, size=(n_customers, 2)).astype(np.float32)
    demands = rng.randint(1, 10, size=n_customers).astype(np.float32)
    chargers = rng.uniform(0.1, 0.9, size=(n_chargers, 2)).astype(np.float32)

    distances_from_depot = np.linalg.norm(customers - depot, axis=1)
    tw_start = (distances_from_depot * 0.2).astype(np.float32)
    tw_end = (tw_start + 2.5 + rng.uniform(0, 1.0, size=n_customers)).astype(np.float32)

    return {
        "depot": depot,
        "customers": customers,
        "demands": demands,
        "chargers": chargers,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "cargo_capacity": float(cargo_capacity),
        "battery_capacity": float(battery_capacity),
    }


def pairwise_dist(coords):
    coords = np.asarray(coords, dtype=np.float32)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float32)


# ============================================================================
# INSTANCE BUILDING
# ============================================================================
#
# We build instances with the keys most likely expected by rl4evrp.
# Since your config says feature schema is based on:
# x, y, demand_normalized, is_charger, is_depot, cargo_capacity_norm, battery_capacity_norm
# the env likely derives features internally from coords/node_types/demands/cargo_cap/battery_cap.
# ============================================================================

def make_instance_dict(coords, node_types, demands, cargo_cap, battery_cap, tw_start=None, tw_end=None):
    coords = np.asarray(coords, dtype=np.float32)
    node_types = np.asarray(node_types, dtype=np.int64)
    demands = np.asarray(demands, dtype=np.float32)

    n = len(coords)
    if tw_start is None:
        tw_start = np.zeros(n, dtype=np.float32)
    if tw_end is None:
        tw_end = np.full(n, 999.0, dtype=np.float32)

    D = pairwise_dist(coords)

    inst = {
        # geometry
        "coords": coords,
        "locs": coords,
        "locations": coords,

        # node definitions
        "node_types": node_types,
        "demands": demands,
        "demand": demands,

        # capacities: use the names your env is likely expecting
        "cargo_cap": float(cargo_cap),
        "cargo_capacity": float(cargo_cap),
        "battery_cap": float(battery_cap),
        "battery_capacity": float(battery_cap),

        # optional support keys
        "distance_matrix": D,
        "dist_matrix": D,
        "D": D,

        # EVRPTW support
        "tw_start": np.asarray(tw_start, dtype=np.float32),
        "tw_end": np.asarray(tw_end, dtype=np.float32),

        # metadata
        "n_nodes": int(n),
        "n_customers": int(np.sum(node_types == 1)),
        "n_chargers": int(np.sum(node_types == 2)),
        "depot_idx": 0,
    }
    return inst


def build_variant_instance(shared, variant):
    depot = shared["depot"].reshape(1, 2)
    customers = shared["customers"]
    demands = shared["demands"]
    chargers = shared["chargers"]

    cargo_cap = shared["cargo_capacity"]
    battery_cap = shared["battery_capacity"]

    if variant == "CVRP":
        coords = np.vstack([depot, customers])
        node_types = np.array([0] + [1] * len(customers), dtype=np.int64)
        all_demands = np.concatenate([[0.0], demands]).astype(np.float32)

        # Keep battery large since the model expects EV features
        return make_instance_dict(
            coords=coords,
            node_types=node_types,
            demands=all_demands,
            cargo_cap=cargo_cap,
            battery_cap=battery_cap,
            tw_start=np.zeros(len(coords), dtype=np.float32),
            tw_end=np.full(len(coords), 999.0, dtype=np.float32),
        )

    elif variant == "EVRP":
        coords = np.vstack([depot, customers, chargers])
        node_types = np.array([0] + [1] * len(customers) + [2] * len(chargers), dtype=np.int64)
        all_demands = np.concatenate([[0.0], demands, np.zeros(len(chargers), dtype=np.float32)])

        return make_instance_dict(
            coords=coords,
            node_types=node_types,
            demands=all_demands,
            cargo_cap=cargo_cap,
            battery_cap=battery_cap,
            tw_start=np.zeros(len(coords), dtype=np.float32),
            tw_end=np.full(len(coords), 999.0, dtype=np.float32),
        )

    elif variant == "EVRPTW":
        coords = np.vstack([depot, customers, chargers])
        node_types = np.array([0] + [1] * len(customers) + [2] * len(chargers), dtype=np.int64)
        all_demands = np.concatenate([[0.0], demands, np.zeros(len(chargers), dtype=np.float32)])

        tw_start = np.concatenate([
            [0.0],
            shared["tw_start"],
            np.zeros(len(chargers), dtype=np.float32)
        ]).astype(np.float32)

        tw_end = np.concatenate([
            [999.0],
            shared["tw_end"],
            np.full(len(chargers), 999.0, dtype=np.float32)
        ]).astype(np.float32)

        return make_instance_dict(
            coords=coords,
            node_types=node_types,
            demands=all_demands,
            cargo_cap=cargo_cap,
            battery_cap=battery_cap,
            tw_start=tw_start,
            tw_end=tw_end,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


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
# RUN VARIANT
# ============================================================================

def solve_variant(model, framework, inst, variant_name, greedy=True):
    reward, route, dist, info, _, _, env = run_episode(
        model,
        inst,
        device=str(framework.device),
        greedy=greedy
    )

    route = list(route) if not isinstance(route, list) else route
    coords = inst["coords"]
    node_types = inst["node_types"]
    demands = inst["demands"]

    customer_nodes = set(np.where(node_types == 1)[0].tolist())
    served = len(customer_nodes.intersection(set(route)))
    total_cust = len(customer_nodes)

    return {
        "variant": variant_name,
        "route": route,
        "reward": float(reward),
        "total_dist": float(dist),
        "served": int(served),
        "total_cust": int(total_cust),
        "charger_visits": int(info.get("charger_visits", 0)),
        "locs": coords,
        "node_types": node_types,
        "demands": demands,
        "feasible_log": list(getattr(env, "feasible_log", [])),
        "battery_log": list(getattr(env, "battery_log", [])),
        "info": info,
    }


# ============================================================================
# PLOT
# ============================================================================

def plot_figure(results, output_path):
    n = len(results)
    has_logs = any(len(r["feasible_log"]) > 0 for r in results)

    if has_logs:
        fig, axes = plt.subplots(
            2, n,
            figsize=(4.8 * n, 7.5),
            gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.30, "wspace": 0.25}
        )
        if n == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.5))
        if n == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array([axes])

    c_depot = "#c0392b"
    c_customer = "#3498db"
    c_charger = "#27ae60"
    c_route = "#2c3e50"
    variant_line_colors = ["#888780", "#27ae60", "#c0392b", "#8e44ad"]

    for col, r in enumerate(results):
        ax = axes[0, col]
        locs = r["locs"]
        node_types = r["node_types"]
        demands = r["demands"]
        route = r["route"]

        # ---- Route arrows ----
        for i in range(len(route) - 1):
            n1, n2 = route[i], route[i + 1]

            if n2 == 0:
                col_e = "red"
            elif node_types[n2] == 2:
                col_e = "#2ECC71"
            else:
                col_e = "#3498DB"

            ax.annotate(
                "",
                xy=locs[n2],
                xytext=locs[n1],
                arrowprops=dict(
                    arrowstyle="->",
                    color=col_e,
                    lw=2,
                    alpha=0.7,
                    shrinkA=6,
                    shrinkB=6
                ),
                zorder=2
            )

        # ---- Nodes ----
        for n in range(len(locs)):
            x, y = locs[n]

            if n == 0:
                ax.plot(x, y, "*", color="red", ms=22, zorder=6)
                ax.annotate(
                    "Depot",
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=9,
                    color="red",
                    fontweight="bold"
                )

            elif node_types[n] == 1:
                demand_val = float(demands[n]) if n < len(demands) else 0.0
                sz = 80 + demand_val * 20

                ax.scatter(
                    x, y,
                    s=sz,
                    color="#3498DB",
                    zorder=5,
                    edgecolors="white",
                    linewidths=1.5
                )
                ax.annotate(
                    f"C{n}\n({demand_val:.0f})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color="#2980B9"
                )

            else:
                ax.plot(x, y, "s", color="#2ECC71", ms=13, zorder=5)
                ax.annotate(
                    f"CH{n}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color="#27AE60"
                )

        # ---- Route step labels ----
        max_labels = min(len(route), 25)
        for i, n in enumerate(route[:max_labels]):
            ax.text(
                locs[n, 0] + 0.02,
                locs[n, 1] + 0.03,
                str(i),
                fontsize=7,
                color="gray",
                ha="center"
            )

        customers = np.where(node_types == 1)[0]
        n_served = sum(1 for n in route if node_types[n] == 1)

        ax.set_xlim(-0.05, 1.10)
        ax.set_ylim(-0.05, 1.10)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        if col == 0:
            ax.set_ylabel("Y")

        ax.grid(alpha=0.2)

        extra = ""
        if "batt_violations" in r["info"]:
            extra += f"   batt_viol={r['info']['batt_violations']}"
        if "cargo_violations" in r["info"]:
            extra += f"   cargo_viol={r['info']['cargo_violations']}"

        ax.set_title(
            f"{r['variant']} | Total dist={r['total_dist']:.3f}\n"
            f"Served={n_served}/{len(customers)}  Chargers={r['charger_visits']}{extra}",
            fontsize=10,
            fontweight="bold"
        )

        if col == 0:
            legend_els = [
                Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                       markeredgecolor="red", markersize=14, label="Depot"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB",
                       markeredgecolor="white", markersize=9, label="Customer (size=demand)"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ECC71",
                       markeredgecolor="#2ECC71", markersize=9, label="Charger"),
            ]
            ax.legend(handles=legend_els, fontsize=8, loc="lower right", framealpha=0.9)

        extra = ""
        if "batt_violations" in r["info"]:
            extra += f"   batt_viol={r['info']['batt_violations']}"
        if "cargo_violations" in r["info"]:
            extra += f"   cargo_viol={r['info']['cargo_violations']}"

        ax.set_title(
            f"{r['variant']}\n"
            f"cost={r['total_dist']:.2f}   served={r['served']}/{r['total_cust']}   "
            f"chargers={r['charger_visits']}{extra}",
            fontsize=10,
            fontweight="bold"
        )

        if has_logs:
            ax = axes[1, col]
            steps = range(len(r["feasible_log"]))
            color = variant_line_colors[col % len(variant_line_colors)]

            if len(r["feasible_log"]) > 0:
                ax.plot(steps, r["feasible_log"], color=color, linewidth=1.8, alpha=0.85)
                ax.fill_between(steps, r["feasible_log"], color=color, alpha=0.1)

            if len(r["battery_log"]) > 0 and any(b != 1.0 for b in r["battery_log"]):
                ax2 = ax.twinx()
                ax2.plot(steps, r["battery_log"], color=c_depot, linewidth=1.2,
                         alpha=0.6, linestyle="--")
                ax2.set_ylim(-0.05, 1.1)
                ax2.set_ylabel("Battery", fontsize=8, color=c_depot, alpha=0.7)
                ax2.tick_params(axis="y", labelcolor=c_depot, labelsize=7)

            ax.set_xlabel("Decision step", fontsize=9)
            if col == 0:
                ax.set_ylabel("Feasible actions", fontsize=10)
            ax.set_title("Masking dynamics", fontsize=10, fontweight="bold", loc="left")
            ax.grid(axis="y", alpha=0.12)
            ax.set_ylim(bottom=0)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved figure to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--output", default="variant_comparison_output.pdf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-customers", type=int, default=15)
    parser.add_argument("--n-chargers", type=int, default=3)
    parser.add_argument("--cargo-cap", type=float, default=30.0)
    parser.add_argument("--battery-cap", type=float, default=100.0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    print("Loading RL4EVRP framework and model...")
    framework, model = load_model(args.model)
    print(f"✓ Model loaded on device: {framework.device}")

    print(f"Generating shared instance with seed={args.seed} ...")
    shared = generate_shared_instance(
        n_customers=args.n_customers,
        n_chargers=args.n_chargers,
        seed=args.seed,
        cargo_capacity=args.cargo_cap,
        battery_capacity=args.battery_cap,
    )

    results = []

    for variant in ["CVRP", "EVRP", "EVRPTW"]:
        print(f"\nRunning variant: {variant}")
        inst = build_variant_instance(shared, variant)

        print("Instance summary:")
        print(f"  n_nodes      = {inst['n_nodes']}")
        print(f"  n_customers  = {inst['n_customers']}")
        print(f"  n_chargers   = {inst['n_chargers']}")
        print(f"  cargo_cap    = {inst['cargo_cap']}")
        print(f"  battery_cap  = {inst['battery_cap']}")

        try:
            res = solve_variant(
                model=model,
                framework=framework,
                inst=inst,
                variant_name=variant,
                greedy=not args.stochastic,
            )
            results.append(res)

            print(f"  reward            = {res['reward']:.4f}")
            print(f"  distance          = {res['total_dist']:.4f}")
            print(f"  served customers  = {res['served']}/{res['total_cust']}")
            print(f"  charger visits    = {res['charger_visits']}")
            print(f"  route             = {res['route']}")
            if "batt_violations" in res["info"]:
                print(f"  batt violations   = {res['info']['batt_violations']}")
            if "cargo_violations" in res["info"]:
                print(f"  cargo violations  = {res['info']['cargo_violations']}")

        except Exception as e:
            print(f"  FAILED on {variant}")
            print(f"  Error: {type(e).__name__}: {e}")

    if not results:
        raise RuntimeError("All variants failed.")

    plot_figure(results, args.output)


if __name__ == "__main__":
    main()