"""
Variant Comparison Figure — RL version

Replace the NN heuristic with your trained RL policy.

Important:
- You must import/rebuild the SAME model class used during training.
- The .pt file you uploaded is a state_dict, not a full model object.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn.functional as F


# ============================================================================
# INSTANCE GENERATION
# ============================================================================

def generate_shared_instance(n_customers=15, n_chargers=3, seed=42):
    rng = np.random.RandomState(seed)

    depot = rng.uniform(0.3, 0.7, size=2)
    customers = rng.uniform(0.05, 0.95, size=(n_customers, 2))
    demands = rng.randint(1, 10, size=n_customers).astype(float)
    charger_locs = rng.uniform(0.1, 0.9, size=(n_chargers, 2))

    distances_from_depot = np.linalg.norm(customers - depot, axis=1)
    tw_start = distances_from_depot * 0.2
    tw_end = tw_start + 2.5 + rng.uniform(0, 1.0, size=n_customers)

    return {
        "depot": depot,
        "customers": customers,
        "demands": demands,
        "charger_locs": charger_locs,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "capacity": 50.0,
        "battery_capacity": 1.2,
        "energy_rate": 1.0,
    }


# ============================================================================
# BASE ENV MIXIN FOR RL
# ============================================================================

class RLFeatureMixin:
    """
    Builds node features for the RL model.

    We use 7 features per node because your checkpoint suggests input_dim = 7.
    You may need to change the exact ordering to match your training code.

    Proposed 7 features:
    0: x
    1: y
    2: normalized demand
    3: node type (0 depot, 1 customer, 2 charger) normalized
    4: visited flag
    5: remaining capacity ratio
    6: battery ratio or time info depending on variant
    """

    def get_action_mask(self):
        mask = np.zeros(self.n_nodes, dtype=bool)
        feasible = self.get_feasible()
        mask[feasible] = True
        return mask

    def get_node_features(self):
        feats = np.zeros((self.n_nodes, 7), dtype=np.float32)

        max_demand = max(1.0, np.max(self.demands))
        rem_cap_ratio = self.remaining_cap / max(1e-8, self.capacity)

        # battery ratio if exists, else 1.0
        if hasattr(self, "battery_cap"):
            battery_ratio = self.battery / max(1e-8, self.battery_cap)
        else:
            battery_ratio = 1.0

        for i in range(self.n_nodes):
            x, y = self.locs[i]
            visited = 1.0 if i in self.visited else 0.0

            feats[i, 0] = x
            feats[i, 1] = y
            feats[i, 2] = self.demands[i] / max_demand
            feats[i, 3] = self.node_types[i] / 2.0
            feats[i, 4] = visited
            feats[i, 5] = rem_cap_ratio
            feats[i, 6] = battery_ratio

        return feats

    def get_context_features(self):
        """
        Global/context features used by decoder if needed.
        Your checkpoint suggests decoder.proj_ctx takes 130 dims.
        That often means 128 encoder context + 2 extra features.

        We expose two extras here:
        - remaining capacity ratio
        - battery ratio (or 1.0 for CVRP)

        If your original model expects something else, replace this.
        """
        rem_cap_ratio = self.remaining_cap / max(1e-8, self.capacity)

        if hasattr(self, "battery_cap"):
            battery_ratio = self.battery / max(1e-8, self.battery_cap)
        else:
            battery_ratio = 1.0

        return np.array([rem_cap_ratio, battery_ratio], dtype=np.float32)


# ============================================================================
# ENVIRONMENTS
# ============================================================================

class CVRPEnv(RLFeatureMixin):
    def __init__(self, instance):
        n_cust = len(instance["customers"])
        self.locs = np.vstack([instance["depot"].reshape(1, 2), instance["customers"]])
        self.n_nodes = n_cust + 1
        self.demands = np.concatenate([[0], instance["demands"]])
        self.capacity = instance["capacity"]
        self.node_types = np.array([0] + [1] * n_cust)  # 0=depot, 1=customer
        self.D = np.linalg.norm(self.locs[:, None] - self.locs[None, :], axis=-1)
        self.reset()

    def reset(self):
        self.cur = 0
        self.visited = set()
        self.remaining_cap = self.capacity
        self.total_dist = 0.0
        self.route = [0]
        self.feasible_log = []
        self.battery_log = []
        return self

    def get_feasible(self):
        feasible = []
        n_cust = (self.node_types == 1).sum()

        for j in range(self.n_nodes):
            if self.node_types[j] == 1:
                if j in self.visited:
                    continue
                if self.demands[j] > self.remaining_cap:
                    continue
            feasible.append(j)

        if len(self.visited) >= n_cust:
            feasible = [0]

        return feasible

    def step(self, action):
        self.total_dist += self.D[self.cur, action]
        self.route.append(action)

        if action == 0:
            self.remaining_cap = self.capacity
        elif self.node_types[action] == 1:
            self.visited.add(action)
            self.remaining_cap -= self.demands[action]

        self.cur = action


class EVRPEnv(RLFeatureMixin):
    def __init__(self, instance):
        n_cust = len(instance["customers"])
        n_chrg = len(instance["charger_locs"])

        self.locs = np.vstack([
            instance["depot"].reshape(1, 2),
            instance["customers"],
            instance["charger_locs"]
        ])
        self.n_nodes = 1 + n_cust + n_chrg
        self.n_cust = n_cust

        self.demands = np.concatenate([[0], instance["demands"], np.zeros(n_chrg)])
        self.capacity = instance["capacity"]
        self.battery_cap = instance["battery_capacity"]
        self.energy_rate = instance["energy_rate"]

        self.node_types = np.array([0] + [1] * n_cust + [2] * n_chrg)
        self.D = np.linalg.norm(self.locs[:, None] - self.locs[None, :], axis=-1)
        self.reset()

    def reset(self):
        self.cur = 0
        self.visited = set()
        self.remaining_cap = self.capacity
        self.battery = self.battery_cap
        self.total_dist = 0.0
        self.route = [0]
        self.feasible_log = []
        self.battery_log = []
        self.charger_visits = 0
        return self

    def can_reach_safety(self, from_node, battery_after):
        if battery_after >= self.D[from_node, 0] * self.energy_rate:
            return True
        for j in range(self.n_nodes):
            if self.node_types[j] == 2:
                if battery_after >= self.D[from_node, j] * self.energy_rate:
                    return True
        return False

    def get_feasible(self):
        feasible = []

        for j in range(self.n_nodes):
            if j == self.cur:
                continue
            if self.node_types[j] == 1 and j in self.visited:
                continue
            if self.node_types[j] == 1 and self.demands[j] > self.remaining_cap:
                continue

            energy_needed = self.D[self.cur, j] * self.energy_rate
            if energy_needed > self.battery:
                continue

            battery_after = self.battery - energy_needed
            if not self.can_reach_safety(j, battery_after):
                continue

            feasible.append(j)

        depot_energy = self.D[self.cur, 0] * self.energy_rate
        if 0 not in feasible and depot_energy <= self.battery:
            feasible.append(0)

        if not feasible:
            feasible = [0]

        return feasible

    def step(self, action):
        energy = self.D[self.cur, action] * self.energy_rate
        self.total_dist += self.D[self.cur, action]
        self.battery -= energy
        self.route.append(action)

        if action == 0:
            self.remaining_cap = self.capacity
            self.battery = self.battery_cap
        elif self.node_types[action] == 2:
            self.battery = self.battery_cap
            self.charger_visits += 1
        elif self.node_types[action] == 1:
            self.visited.add(action)
            self.remaining_cap -= self.demands[action]

        self.cur = action


class EVRPTWEnv(EVRPEnv):
    def __init__(self, instance):
        super().__init__(instance)

        self.tw_start = np.concatenate([
            [0],
            instance["tw_start"],
            np.zeros(len(instance["charger_locs"]))
        ])
        self.tw_end = np.concatenate([
            [999],
            instance["tw_end"],
            np.full(len(instance["charger_locs"]), 999.0)
        ])

        self.speed = 1.0
        self.current_time = 0.0

    def reset(self):
        super().reset()
        self.current_time = 0.0
        return self

    def get_feasible(self):
        base_feasible = super().get_feasible()
        tw_feasible = []

        for j in base_feasible:
            travel_time = self.D[self.cur, j] / self.speed
            arrival = self.current_time + travel_time
            if arrival <= self.tw_end[j]:
                tw_feasible.append(j)

        if not tw_feasible:
            tw_feasible = [0]

        return tw_feasible

    def step(self, action):
        travel_time = self.D[self.cur, action] / self.speed
        self.current_time += travel_time

        if self.node_types[action] == 1:
            self.current_time = max(self.current_time, self.tw_start[action])

        super().step(action)

    def get_node_features(self):
        """
        For EVRPTW you may want feature 6 to encode time instead of battery,
        or use both in the original training code.
        Since your checkpoint only supports 7 features, this is a design choice.

        Here I keep battery ratio in feature 6 and inject time-window urgency
        into demand slot only if you want. For now we leave it simple.

        If your EVRPTW model was trained differently, replace this function.
        """
        feats = super().get_node_features()

        # Example optional tweak:
        # encode lateness pressure by modifying an extra signal
        # current time normalized
        cur_t = min(self.current_time / max(1.0, np.max(self.tw_end)), 1.5)

        # overwrite feature 6 with current-time info if you trained that way:
        # feats[:, 6] = cur_t

        return feats
    


# ============================================================================
# RL SOLVER
# ============================================================================

def select_action_from_logits(logits, mask, greedy=True):
    """
    logits: torch tensor [n_nodes]
    mask:   torch bool tensor [n_nodes], True = feasible
    """
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9

    if greedy:
        action = torch.argmax(masked_logits).item()
    else:
        probs = F.softmax(masked_logits, dim=-1)
        action = torch.multinomial(probs, 1).item()

    return action


def solve_with_rl(env, model, device="cpu", greedy=True, max_steps=None):
    """
    Generic RL rollout on any env implementing:
      - get_node_features()
      - get_context_features()
      - get_action_mask()
      - step(action)
    """
    env.reset()

    if max_steps is None:
        max_steps = env.n_nodes * 4

    n_cust = (env.node_types == 1).sum()

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if len(env.visited) >= n_cust:
                break

            node_feats = torch.tensor(
                env.get_node_features(),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)   # [1, n_nodes, feat_dim]

            ctx_feats = torch.tensor(
                env.get_context_features(),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)   # [1, ctx_dim]

            mask = torch.tensor(
                env.get_action_mask(),
                dtype=torch.bool,
                device=device
            ).unsqueeze(0)   # [1, n_nodes]

            env.feasible_log.append(int(mask.sum().item()))

            if hasattr(env, "battery_cap"):
                env.battery_log.append(env.battery / env.battery_cap)
            else:
                env.battery_log.append(1.0)

            # -------------------------------------------------------------
            # IMPORTANT:
            # Replace this model call by the exact forward used in training.
            # -------------------------------------------------------------
            logits = forward_model_for_inference(model, node_feats, ctx_feats, mask)
            # logits shape expected: [1, n_nodes]

            action = select_action_from_logits(logits[0], mask[0], greedy=greedy)
            env.step(action)

        if env.cur != 0:
            env.feasible_log.append(1)
            if hasattr(env, "battery_cap"):
                env.battery_log.append(env.battery / env.battery_cap)
            else:
                env.battery_log.append(1.0)
            env.step(0)

    return env


# ============================================================================
# MODEL LOADING
# ============================================================================

def build_model_for_checkpoint():
    """
    Recreate the SAME architecture used in training.

    Your checkpoint suggests roughly:
    - input_dim = 7
    - hidden_dim = 128
    - n_encoder_layers = 3

    But you must use your exact original class.
    """
    # Example only:
    # from my_training_code import ActorCriticModel
    # model = ActorCriticModel(
    #     input_dim=7,
    #     hidden_dim=128,
    #     n_layers=3,
    # )
    # return model

    raise NotImplementedError(
        "Import your original model class here and instantiate it."
    )


def load_rl_model(checkpoint_path, device="cpu"):
    model = build_model_for_checkpoint()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def forward_model_for_inference(model, node_feats, ctx_feats, mask):
    """
    Convert generic env state into model logits.

    You MUST adapt this to your training code.

    Expected return: logits of shape [B, n_nodes]
    """
    # Example 1:
    # logits, value = model(node_feats, ctx_feats, mask)

    # Example 2:
    # out = model(node_feats, ctx_feats, action_mask=mask)
    # logits = out["logits"]

    # Example 3:
    # logits = model.policy(node_feats, ctx_feats, mask)

    raise NotImplementedError(
        "Replace with your model's actual forward/inference call."
    )



def plot_figure(results, output_path):
    n = len(results)
    fig, axes = plt.subplots(
        2, n, figsize=(4.8 * n, 7.5),
        gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.30, "wspace": 0.25}
    )
    if n == 1:
        axes = axes.reshape(-1, 1)

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

        for i in range(len(route) - 1):
            x0, y0 = locs[route[i]]
            x1, y1 = locs[route[i+1]]
            ax.plot([x0, x1], [y0, y1], color=c_route, linewidth=1.0, alpha=0.5, zorder=2)

        for i in range(len(locs)):
            x, y = locs[i]
            if node_types[i] == 0:
                ax.plot(x, y, "s", color=c_depot, markersize=11, zorder=5,
                        markeredgecolor="white", markeredgewidth=1)
            elif node_types[i] == 2:
                ax.plot(x, y, "^", color=c_charger, markersize=9, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.5)
            else:
                size = 25 + demands[i] * 8
                ax.scatter(x, y, s=size, c=c_customer, zorder=4,
                           edgecolors="white", linewidth=0.5, alpha=0.85)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=9)
        if col == 0:
            ax.set_ylabel("y", fontsize=9)

        cost = r["total_dist"]
        served = r["served"]
        total_cust = r["total_cust"]
        chg = r.get("charger_visits", 0)

        ax.set_title(
            f"{r['variant']}\n"
            f"cost = {cost:.2f}   served = {served}/{total_cust}   chargers = {chg}",
            fontsize=10, fontweight="bold"
        )

        if col == 0:
            legend_els = [
                Line2D([0], [0], marker="s", color="w", markerfacecolor=c_depot, markersize=7, label="Depot"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=c_customer, markersize=7, label="Customer"),
                Line2D([0], [0], marker="^", color="w", markerfacecolor=c_charger, markersize=7, label="Charger"),
            ]
            ax.legend(handles=legend_els, fontsize=7, loc="lower right", framealpha=0.9)

        ax = axes[1, col]
        steps = range(len(r["feasible_log"]))
        color = variant_line_colors[col % len(variant_line_colors)]

        ax.plot(steps, r["feasible_log"], color=color, linewidth=1.8, alpha=0.85)
        ax.fill_between(steps, r["feasible_log"], color=color, alpha=0.1)

        if any(b != 1.0 for b in r["battery_log"]):
            ax2 = ax.twinx()
            ax2.plot(steps, r["battery_log"], color=c_depot, linewidth=1.2,
                     alpha=0.6, linestyle="--")
            ax2.set_ylim(-0.05, 1.1)
            ax2.set_ylabel("Battery", fontsize=8, color=c_depot, alpha=0.7)
            ax2.tick_params(axis="y", labelcolor=c_depot, labelsize=7)
            ax2.axhline(0.3, color=c_depot, ls=":", lw=0.6, alpha=0.3)

        ax.set_xlabel("Decision step", fontsize=9)
        if col == 0:
            ax.set_ylabel("Feasible actions", fontsize=10)
        ax.set_title("Masking dynamics", fontsize=10, fontweight="bold", loc="left")
        ax.grid(axis="y", alpha=0.12)
        ax.set_xlim(0, max(1, len(r["feasible_log"]) - 1))
        ax.set_ylim(bottom=0)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="variant_comparison.pdf")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-customers", type=int, default=15)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    print(f"Generating instance: {args.n_customers} customers, seed={args.seed}")
    instance = generate_shared_instance(args.n_customers, n_chargers=3, seed=args.seed)

    model = load_rl_model(args.checkpoint, device=args.device)

    results = []

    # ---- CVRP ----
    print("Solving CVRP with RL...")
    env = solve_with_rl(CVRPEnv(instance), model, device=args.device, greedy=args.greedy)
    n_cust = (env.node_types == 1).sum()
    results.append({
        "variant": "CVRP",
        "route": env.route,
        "total_dist": env.total_dist,
        "served": len(env.visited),
        "total_cust": n_cust,
        "charger_visits": 0,
        "locs": env.locs,
        "node_types": env.node_types,
        "demands": env.demands,
        "feasible_log": env.feasible_log,
        "battery_log": env.battery_log,
    })
    print(f"  cost={env.total_dist:.2f}, served={len(env.visited)}/{n_cust}")

    # ---- EVRP ----
    print("Solving EVRP with RL...")
    env = solve_with_rl(EVRPEnv(instance), model, device=args.device, greedy=args.greedy)
    n_cust = (env.node_types == 1).sum()
    results.append({
        "variant": "EVRP",
        "route": env.route,
        "total_dist": env.total_dist,
        "served": len(env.visited),
        "total_cust": n_cust,
        "charger_visits": env.charger_visits,
        "locs": env.locs,
        "node_types": env.node_types,
        "demands": env.demands,
        "feasible_log": env.feasible_log,
        "battery_log": env.battery_log,
    })
    print(f"  cost={env.total_dist:.2f}, served={len(env.visited)}/{n_cust}, charger_visits={env.charger_visits}")

    # ---- EVRPTW ----
    print("Solving EVRPTW with RL...")
    env = solve_with_rl(EVRPTWEnv(instance), model, device=args.device, greedy=args.greedy)
    n_cust = (env.node_types == 1).sum()
    results.append({
        "variant": "EVRPTW",
        "route": env.route,
        "total_dist": env.total_dist,
        "served": len(env.visited),
        "total_cust": n_cust,
        "charger_visits": env.charger_visits,
        "locs": env.locs,
        "node_types": env.node_types,
        "demands": env.demands,
        "feasible_log": env.feasible_log,
        "battery_log": env.battery_log,
    })
    print(f"  cost={env.total_dist:.2f}, served={len(env.visited)}/{n_cust}, charger_visits={env.charger_visits}")

    plot_figure(results, args.output)