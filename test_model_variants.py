"""
Test a trained EVRP model on both CVRP and EVRP instances.

Question: Can a model trained on EVRP also solve CVRP?
If yes, this supports the transfer/generalization story.

This script provides TWO modes:

    Mode 1 (standalone): Uses simplified built-in environments
    Mode 2 (framework): Uses YOUR EVRPEnv — uncomment Section A

Usage:
    python test_model_variants.py --checkpoint agent_episode_26800.pt

IMPORTANT: The model was trained with 7 input features on EVRP.
For CVRP, we zero out battery/charger features (features stay 7-dim).
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL RECONSTRUCTION (from your checkpoint structure)
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = k
        B, N, D = q.shape
        H, d = self.num_heads, self.head_dim
        Q = self.W_q(q).view(B, -1, H, d).transpose(1, 2)
        K = self.W_k(k).view(B, -1, H, d).transpose(1, 2)
        V = self.W_v(v).view(B, -1, H, d).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (d ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, D)
        return self.W_o(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=128, ff_dim=256, num_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(),
                                nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        h, _ = self.attn(x, mask=mask)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, ff_dim=256,
                 num_layers=3, num_heads=8, context_extra=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_embed = nn.Linear(input_dim, embed_dim)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, ff_dim, num_heads) for _ in range(num_layers)
        ])
        self.decoder_proj_ctx = nn.Linear(embed_dim + context_extra, embed_dim)
        self.decoder_cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, node_features, context, mask=None):
        h = self.encoder_embed(node_features)
        for layer in self.encoder_layers:
            h = layer(h)
        ctx = self.decoder_proj_ctx(context)
        glimpse, attn = self.decoder_cross_attn(ctx, h, h)
        logits = (glimpse @ h.transpose(-2, -1)).squeeze(1)
        logits = 10.0 * torch.tanh(logits)
        value = self.value_head(h.mean(dim=1))
        return logits, value, h, attn


def load_model(checkpoint_path, device="cpu"):
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Remap keys from checkpoint format to our module format
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("encoder.embed.", "encoder_embed.")
        nk = nk.replace("encoder.layers.", "encoder_layers.")
        nk = nk.replace("decoder.proj_ctx.", "decoder_proj_ctx.")
        nk = nk.replace("decoder.cross_attn.", "decoder_cross_attn.")
        new_sd[nk] = v

    input_dim = sd["encoder.embed.weight"].shape[1]
    embed_dim = sd["encoder.embed.weight"].shape[0]
    ff_dim = sd["encoder.layers.0.ff.0.weight"].shape[0]
    ctx_extra = sd["decoder.proj_ctx.weight"].shape[1] - embed_dim
    num_layers = len(set(k.split(".")[2] for k in sd if k.startswith("encoder.layers")))

    model = PolicyNetwork(input_dim, embed_dim, ff_dim, num_layers, context_extra=ctx_extra)
    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()
    print(f"Loaded model: {input_dim}→{embed_dim}d, {num_layers} layers, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


# ============================================================================
# ENVIRONMENT
# ============================================================================

class SimpleEnv:
    """
    Minimal CVRP/EVRP environment for testing.
    Produces 7-dim features matching the model's expected input.

    ADAPT: Replace build_features() and get_context() to match
    your actual training code's feature construction.
    """
    def __init__(self, n_customers=20, n_chargers=3, capacity=30.0,
                 battery_capacity=100.0, energy_rate=1.0,
                 variant="evrp", seed=42):
        rng = np.random.RandomState(seed)
        self.variant = variant
        self.capacity = capacity
        self.battery_cap = battery_capacity
        self.energy_rate = energy_rate

        # Generate locations
        depot = rng.uniform(0.2, 0.8, size=2)
        customers = rng.uniform(0, 1, size=(n_customers, 2))
        self.demands_raw = rng.randint(1, 10, size=n_customers).astype(float)

        # Chargers: use charger_prob=0.15 style — ~15% of nodes become chargers
        if variant in ("evrp", "evrptw"):
            n_chargers = max(1, int(n_customers * 0.15))
            chargers = rng.uniform(0, 1, size=(n_chargers, 2))
            self.locs = np.vstack([depot.reshape(1, 2), customers, chargers])
            self.node_types = np.array([0] + [1]*n_customers + [2]*n_chargers)
            self.demands = np.concatenate([[0], self.demands_raw, np.zeros(n_chargers)])
        else:
            self.locs = np.vstack([depot.reshape(1, 2), customers])
            self.node_types = np.array([0] + [1]*n_customers)
            self.demands = np.concatenate([[0], self.demands_raw])

        self.n_nodes = len(self.locs)
        self.n_customers = n_customers
        self.D = np.linalg.norm(self.locs[:, None] - self.locs[None, :], axis=-1)
        self.reset()

    def reset(self):
        self.cur = 0
        self.visited = set()
        self.remaining_cap = self.capacity
        self.battery = self.battery_cap
        self.total_dist = 0
        self.route = [0]
        self.step_count = 0
        self.charger_visits = 0
        return self

    def get_feasible_mask(self):
        """Returns boolean mask: True = feasible."""
        mask = np.zeros(self.n_nodes, dtype=bool)
        for j in range(self.n_nodes):
            if j == self.cur:
                continue
            if self.node_types[j] == 1:  # customer
                if j in self.visited:
                    continue
                if self.demands[j] > self.remaining_cap:
                    continue
            if self.variant in ("evrp", "evrptw"):
                energy = self.D[self.cur, j] * self.energy_rate
                if energy > self.battery:
                    continue
                # Can return to safety after?
                batt_after = self.battery - energy
                can_return = batt_after >= self.D[j, 0] * self.energy_rate
                if not can_return:
                    # Check chargers
                    for c in range(self.n_nodes):
                        if self.node_types[c] == 2:
                            if batt_after >= self.D[j, c] * self.energy_rate:
                                can_return = True
                                break
                if not can_return:
                    continue
            mask[j] = True

        # Depot always feasible
        if self.D[self.cur, 0] * self.energy_rate <= self.battery or self.variant == "cvrp":
            mask[0] = True

        if not mask.any():
            mask[0] = True

        return mask

    def step(self, action):
        self.total_dist += self.D[self.cur, action]
        if self.variant in ("evrp", "evrptw"):
            self.battery -= self.D[self.cur, action] * self.energy_rate
        self.route.append(action)
        self.step_count += 1

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
        done = len(self.visited) >= self.n_customers
        return done

    def build_features(self):
        """
        Build 7-dim features matching the YAML config exactly:
            0: x                    — normalized x-coordinate
            1: y                    — normalized y-coordinate
            2: demand_normalized    — customer demand / cargo capacity
            3: is_charger           — binary indicator for charger nodes
            4: is_depot             — binary indicator for depot node
            5: cargo_capacity_norm  — remaining cargo / cargo capacity
            6: battery_capacity_norm — remaining battery / battery capacity
        """
        feats = np.zeros((self.n_nodes, 7))
        for i in range(self.n_nodes):
            feats[i, 0] = self.locs[i, 0]                          # x
            feats[i, 1] = self.locs[i, 1]                          # y
            feats[i, 2] = self.demands[i] / self.capacity           # demand_normalized
            feats[i, 3] = 1.0 if self.node_types[i] == 2 else 0.0  # is_charger
            feats[i, 4] = 1.0 if self.node_types[i] == 0 else 0.0  # is_depot
            feats[i, 5] = self.remaining_cap / self.capacity        # cargo_capacity_norm
            feats[i, 6] = self.battery / self.battery_cap           # battery_capacity_norm
        return feats

    def build_context(self, embeddings):
        """
        Build context vector (embed_dim + 2 extra features).
        ADAPT: match your training code's context construction.
        """
        B = embeddings.shape[0]
        cur_embed = embeddings[:, self.cur:self.cur+1, :]
        extra = torch.zeros(B, 1, 2, device=embeddings.device)
        extra[:, 0, 0] = self.remaining_cap / self.capacity
        extra[:, 0, 1] = self.battery / self.battery_cap
        return torch.cat([cur_embed, extra], dim=-1)


# ============================================================================
# SOLVE
# ============================================================================

def solve_with_model(model, env, device="cpu", max_steps=100):
    """Greedy decode using the model."""
    env.reset()

    with torch.no_grad():
        for _ in range(max_steps):
            if len(env.visited) >= env.n_customers:
                break

            feats = env.build_features()
            feats_t = torch.FloatTensor(feats).unsqueeze(0).to(device)

            # Encode
            h = model.encoder_embed(feats_t)
            for layer in model.encoder_layers:
                h = layer(h)

            # Build context
            context = env.build_context(h)

            # Decode
            ctx_proj = model.decoder_proj_ctx(context)
            glimpse, _ = model.decoder_cross_attn(ctx_proj, h, h)
            logits = (glimpse @ h.transpose(-2, -1)).squeeze(1)
            # Use raw logits — do NOT apply tanh clipping
            # The model's actual decoder may scale differently

            # Mask
            mask = env.get_feasible_mask()
            logits_np = logits.squeeze(0).cpu().numpy()
            logits_np[~mask] = float('-inf')

            action = int(np.argmax(logits_np))
            done = env.step(action)

    if env.cur != 0:
        env.step(0)

    return env


def solve_with_nn(env, max_steps=100):
    """Nearest-neighbor baseline."""
    env.reset()
    for _ in range(max_steps):
        if len(env.visited) >= env.n_customers:
            break
        mask = env.get_feasible_mask()
        feasible = np.where(mask)[0]
        # Prefer customers over chargers/depot
        customers = [j for j in feasible if env.node_types[j] == 1]
        if customers:
            dists = [env.D[env.cur, j] for j in customers]
            action = customers[int(np.argmin(dists))]
        elif any(env.node_types[j] == 2 for j in feasible):
            chargers = [j for j in feasible if env.node_types[j] == 2]
            dists = [env.D[env.cur, j] for j in chargers]
            action = chargers[int(np.argmin(dists))]
        else:
            action = 0
        env.step(action)
    if env.cur != 0:
        env.step(0)
    return env


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, variant, n_instances=100, n_customers=15, device="cpu"):
    """Evaluate model and NN on the same instances."""
    model_costs = []
    nn_costs = []
    model_served = []
    nn_served = []
    model_chargers = []

    for seed in range(n_instances):
        # Model
        env = SimpleEnv(n_customers=n_customers, variant=variant, seed=seed+500)
        solve_with_model(model, env, device)
        model_costs.append(env.total_dist)
        model_served.append(len(env.visited))
        model_chargers.append(env.charger_visits)

        # NN baseline (same instance)
        env2 = SimpleEnv(n_customers=n_customers, variant=variant, seed=seed+500)
        solve_with_nn(env2)
        nn_costs.append(env2.total_dist)
        nn_served.append(len(env2.visited))

    model_costs = np.array(model_costs)
    nn_costs = np.array(nn_costs)
    model_served = np.array(model_served)
    nn_served = np.array(nn_served)

    return {
        "model_cost_mean": model_costs.mean(),
        "model_cost_std": model_costs.std(),
        "nn_cost_mean": nn_costs.mean(),
        "nn_cost_std": nn_costs.std(),
        "model_served_mean": model_served.mean(),
        "nn_served_mean": nn_served.mean(),
        "model_feasible_rate": (model_served == n_customers).mean(),
        "nn_feasible_rate": (nn_served == n_customers).mean(),
        "gap_vs_nn": (model_costs.mean() - nn_costs.mean()) / nn_costs.mean() * 100,
        "model_wins": (model_costs < nn_costs).mean() * 100,
        "model_chargers_mean": np.array(model_chargers).mean(),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--n-customers", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    print(f"\nEvaluating on {args.n_instances} instances, {args.n_customers} customers")
    print("=" * 75)

    results = {}
    for variant in ["cvrp", "evrp"]:
        print(f"\n--- {variant.upper()} ---")
        t0 = time.time()
        r = evaluate(model, variant, args.n_instances, args.n_customers, args.device)
        elapsed = time.time() - t0
        results[variant] = r

        print(f"  Model:  cost={r['model_cost_mean']:.3f} ± {r['model_cost_std']:.3f}  "
              f"served={r['model_served_mean']:.1f}/{args.n_customers}  "
              f"feasible={r['model_feasible_rate']:.0%}  "
              f"chargers={r['model_chargers_mean']:.1f}")
        print(f"  NN:     cost={r['nn_cost_mean']:.3f} ± {r['nn_cost_std']:.3f}  "
              f"served={r['nn_served_mean']:.1f}/{args.n_customers}  "
              f"feasible={r['nn_feasible_rate']:.0%}")
        print(f"  Gap:    {r['gap_vs_nn']:+.1f}%  |  Model wins: {r['model_wins']:.0f}%")
        print(f"  Time:   {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*75}")
    print(f"SUMMARY: Can an EVRP-trained model solve CVRP?")
    print(f"{'='*75}")
    print(f"{'Variant':<10} {'Model cost':>12} {'NN cost':>12} {'Gap':>8} "
          f"{'Model wins':>12} {'Feasible':>10}")
    print("-" * 70)
    for variant in ["cvrp", "evrp"]:
        r = results[variant]
        print(f"{variant.upper():<10} {r['model_cost_mean']:>12.3f} "
              f"{r['nn_cost_mean']:>12.3f} {r['gap_vs_nn']:>+7.1f}% "
              f"{r['model_wins']:>11.0f}% {r['model_feasible_rate']:>9.0%}")

    print()
    cvrp_gap = results["cvrp"]["gap_vs_nn"]
    evrp_gap = results["evrp"]["gap_vs_nn"]

    if cvrp_gap < 5 and evrp_gap < 5:
        print("VERDICT: Model generalizes well to both CVRP and EVRP.")
        print("This supports the compositional transfer story.")
    elif evrp_gap < cvrp_gap:
        print("VERDICT: Model performs better on EVRP (trained domain) than CVRP.")
        print("Transfer to CVRP is partial — fine-tuning may help.")
    else:
        print("VERDICT: Results inconclusive — see detailed numbers above.")

    if results["cvrp"]["model_feasible_rate"] < 0.5 or results["evrp"]["model_feasible_rate"] < 0.5:
        print("\n⚠ Low feasibility rate detected. The model may need more training,")
        print("  or the feature construction doesn't match your training code.")
        print("  CRITICAL: Adapt build_features() and build_context() in the script.")
