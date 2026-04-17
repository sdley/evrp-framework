"""
Sanity check: does your trained model beat nearest-neighbor?

Usage:
    python sanity_check.py --checkpoint agent_episode_800.pt --variant cvrp --size 20

This script:
    1. Generates random CVRP/EVRP test instances
    2. Solves them with nearest-neighbor heuristic
    3. Solves them with your trained model (greedy decoding)
    4. Compares costs and prints a clear verdict

You MUST adapt Section A below to match your model class and environment.
Everything else should work out of the box.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================================
# SECTION A: YOUR MODEL AND ENVIRONMENT (ADAPT THIS)
# ============================================================================

# ---- Option 1: Import from your framework ----
# Uncomment and modify these lines to use your actual model class:
#
# from your_framework.models import YourPolicyNetwork
# from your_framework.envs import YourCVRPEnv
#
# def load_your_model(checkpoint_path, device="cpu"):
#     model = YourPolicyNetwork(input_dim=7, embed_dim=128, num_layers=3, ...)
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model
#
# def solve_with_model(model, instances, device="cpu"):
#     """
#     Run greedy decoding on a batch of instances.
#     Returns: costs (list of floats), routes (list of node lists)
#     """
#     costs = []
#     routes = []
#     for inst in instances:
#         env = YourCVRPEnv(inst)
#         state = env.reset()
#         route = []
#         with torch.no_grad():
#             while not state.done:
#                 logits = model(state)
#                 mask = state.get_mask()
#                 logits[mask == 0] = float('-inf')
#                 action = logits.argmax()
#                 state = env.step(action)
#                 route.append(action.item())
#         costs.append(env.get_cost())
#         routes.append(route)
#     return costs, routes


# ---- Option 2: Reconstructed model from your checkpoint structure ----
# This is built from inspecting your state_dict. Verify it matches your code.

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
        if k is None:
            k = q
        if v is None:
            v = k
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
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        h, _ = self.attn(x, mask=mask)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, ff_dim=256,
                 num_layers=3, num_heads=8):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, ff_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, mask)
        return h


class Decoder(nn.Module):
    def __init__(self, embed_dim=128, context_extra=2, num_heads=8):
        super().__init__()
        self.proj_ctx = nn.Linear(embed_dim + context_extra, embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)

    def forward(self, node_embeddings, context, mask=None):
        """
        node_embeddings: (B, N, D) from encoder
        context: (B, 1, D + context_extra) — current state context
        mask: (B, N) — 1 = feasible, 0 = infeasible
        Returns: logits (B, N)
        """
        ctx = self.proj_ctx(context)  # (B, 1, D)
        glimpse, _ = self.cross_attn(ctx, node_embeddings, node_embeddings)
        # Compute logits as dot product
        logits = (glimpse @ node_embeddings.transpose(-2, -1)).squeeze(1)
        # Apply clipping (common in AM-style models)
        logits = 10.0 * torch.tanh(logits)
        return logits


class PolicyNetwork(nn.Module):
    """
    Reconstructed from your checkpoint structure.
    VERIFY this matches your actual model architecture!
    """
    def __init__(self, input_dim=7, embed_dim=128, ff_dim=256,
                 num_layers=3, num_heads=8, context_extra=2):
        super().__init__()
        self.encoder = Encoder(input_dim, embed_dim, ff_dim,
                               num_layers, num_heads)
        self.decoder = Decoder(embed_dim, context_extra, num_heads)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, node_features, context, mask=None):
        embeddings = self.encoder(node_features)
        logits = self.decoder(embeddings, context, mask)
        value = self.value_head(embeddings.mean(dim=1))
        return logits, value, embeddings


def load_your_model(checkpoint_path, device="cpu"):
    """Load model from state_dict checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    # Infer architecture from state_dict
    input_dim = state_dict["encoder.embed.weight"].shape[1]
    embed_dim = state_dict["encoder.embed.weight"].shape[0]
    ff_dim = state_dict["encoder.layers.0.ff.0.weight"].shape[0]
    ctx_input = state_dict["decoder.proj_ctx.weight"].shape[1]
    context_extra = ctx_input - embed_dim
    num_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith("encoder.layers")
    ))

    print(f"Inferred architecture: input={input_dim}, embed={embed_dim}, "
          f"ff={ff_dim}, layers={num_layers}, ctx_extra={context_extra}")

    model = PolicyNetwork(
        input_dim=input_dim, embed_dim=embed_dim, ff_dim=ff_dim,
        num_layers=num_layers, context_extra=context_extra,
    )

    # Try loading — this will fail if architecture doesn't match
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


# ============================================================================
# SECTION B: TEST INSTANCE GENERATION
# ============================================================================

@dataclass
class CVRPInstance:
    """A single CVRP instance."""
    depot: np.ndarray        # (2,) — depot coordinates
    customers: np.ndarray    # (n, 2) — customer coordinates
    demands: np.ndarray      # (n,) — customer demands
    capacity: float          # vehicle capacity

    @property
    def num_customers(self):
        return len(self.customers)

    @property
    def all_locations(self):
        """All locations: depot (index 0) + customers (indices 1..n)."""
        return np.vstack([self.depot.reshape(1, 2), self.customers])

    def distance(self, i, j):
        """Distance between location i and location j (0=depot)."""
        locs = self.all_locations
        return np.linalg.norm(locs[i] - locs[j])

    def distance_matrix(self):
        locs = self.all_locations
        return np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)


def generate_cvrp_instances(num_instances: int, num_customers: int,
                            capacity: float = 50.0, seed: int = 42):
    """Generate random CVRP instances."""
    rng = np.random.RandomState(seed)
    instances = []
    for _ in range(num_instances):
        depot = rng.uniform(0, 1, size=2)
        customers = rng.uniform(0, 1, size=(num_customers, 2))
        demands = rng.randint(1, 10, size=num_customers).astype(float)
        instances.append(CVRPInstance(depot, customers, demands, capacity))
    return instances


# ============================================================================
# SECTION C: NEAREST-NEIGHBOR HEURISTIC
# ============================================================================

def nearest_neighbor_cvrp(instance: CVRPInstance):
    """
    Nearest-neighbor heuristic for CVRP.
    At each step, go to the nearest unvisited customer that fits in the vehicle.
    Return to depot when no customer fits.
    """
    n = instance.num_customers
    dist_mat = instance.distance_matrix()
    visited = [False] * n
    route = [0]  # start at depot
    current = 0
    remaining_cap = instance.capacity
    total_cost = 0.0

    for _ in range(n):
        best_next = -1
        best_dist = float('inf')

        # Find nearest feasible unvisited customer
        for j in range(n):
            if visited[j]:
                continue
            cust_idx = j + 1  # node index (0 is depot)
            if instance.demands[j] <= remaining_cap:
                d = dist_mat[current][cust_idx]
                if d < best_dist:
                    best_dist = d
                    best_next = j

        if best_next == -1:
            # No feasible customer — return to depot, start new route
            total_cost += dist_mat[current][0]
            route.append(0)
            current = 0
            remaining_cap = instance.capacity
            # Retry: find nearest from depot
            best_dist = float('inf')
            for j in range(n):
                if visited[j]:
                    continue
                cust_idx = j + 1
                if instance.demands[j] <= remaining_cap:
                    d = dist_mat[current][cust_idx]
                    if d < best_dist:
                        best_dist = d
                        best_next = j

        if best_next == -1:
            break  # all visited

        cust_idx = best_next + 1
        total_cost += dist_mat[current][cust_idx]
        route.append(cust_idx)
        visited[best_next] = True
        remaining_cap -= instance.demands[best_next]
        current = cust_idx

    # Return to depot
    total_cost += dist_mat[current][0]
    route.append(0)

    return total_cost, route


def greedy_insertion_cvrp(instance: CVRPInstance):
    """
    Slightly smarter baseline: nearest-neighbor with savings consideration.
    Still simple but typically ~10% better than pure nearest-neighbor.
    """
    # Just use nearest-neighbor for now — can be upgraded
    return nearest_neighbor_cvrp(instance)


# ============================================================================
# SECTION D: MODEL INFERENCE
# ============================================================================

def build_node_features(instance: CVRPInstance, current_node: int,
                        remaining_capacity: float, visited: list):
    """
    Build the 7-dimensional node features your model expects.

    *** THIS IS THE MOST IMPORTANT FUNCTION TO GET RIGHT ***

    Your model takes 7 features per node. You need to match EXACTLY
    what your training code produces. Common choices:

    Option A: [x, y, demand, is_depot, is_visited, remaining_cap, dist_to_depot]
    Option B: [x, y, demand/capacity, is_depot, visited_flag, current_load, step/n]
    Option C: [x, y, demand, tw_start, tw_end, is_depot, is_visited]

    REPLACE THIS with your actual feature construction.
    """
    n = instance.num_customers
    locs = instance.all_locations  # (n+1, 2)
    num_nodes = n + 1

    features = np.zeros((num_nodes, 7))

    for i in range(num_nodes):
        features[i, 0] = locs[i, 0]  # x
        features[i, 1] = locs[i, 1]  # y

        if i == 0:  # depot
            features[i, 2] = 0.0  # demand
            features[i, 3] = 1.0  # is_depot
        else:
            features[i, 2] = instance.demands[i-1] / instance.capacity  # normalized demand
            features[i, 3] = 0.0  # not depot

        features[i, 4] = 1.0 if (i in visited) else 0.0  # visited flag
        features[i, 5] = remaining_capacity / instance.capacity  # remaining cap
        features[i, 6] = instance.distance(i, 0)  # distance to depot

    return features


def build_context(embeddings: torch.Tensor, current_node: int,
                  remaining_capacity: float, capacity: float):
    """
    Build the 130-dim context vector (128 embedding + 2 extras).

    The 2 extra features are likely:
    - Current node embedding (already in the 128)
    - Remaining capacity (normalized)

    ADAPT THIS to match your training code.
    """
    B = embeddings.shape[0]
    current_embed = embeddings[:, current_node:current_node+1, :]  # (B, 1, 128)

    # The 2 extra context features — guessing remaining capacity + step progress
    extra = torch.zeros(B, 1, 2, device=embeddings.device)
    extra[:, 0, 0] = remaining_capacity / capacity
    extra[:, 0, 1] = 0.0  # adapt as needed

    context = torch.cat([current_embed, extra], dim=-1)  # (B, 1, 130)
    return context


def solve_with_model(model, instances, device="cpu"):
    """
    Run greedy decoding with the trained model on CVRP instances.

    This reconstructs the decode loop from the model architecture.
    VERIFY the feature construction and context building match your training code.
    """
    model.eval()
    costs = []
    routes = []

    with torch.no_grad():
        for inst in instances:
            n = inst.num_customers
            current_node = 0  # start at depot
            remaining_cap = inst.capacity
            visited_set = {0}  # depot always "visited" (but can return)
            visited_customers = set()
            route = [0]
            total_cost = 0.0

            while len(visited_customers) < n:
                # Build features
                node_feats = build_node_features(
                    inst, current_node, remaining_cap, visited_set
                )
                node_tensor = torch.FloatTensor(node_feats).unsqueeze(0).to(device)

                # Encode
                embeddings = model.encoder(node_tensor)

                # Build context
                context = build_context(
                    embeddings, current_node, remaining_cap, inst.capacity
                )

                # Build mask: 1 = feasible, 0 = infeasible
                mask = torch.zeros(1, n + 1, device=device)
                for j in range(n):
                    cust_idx = j + 1
                    if cust_idx not in visited_customers and \
                       inst.demands[j] <= remaining_cap:
                        mask[0, cust_idx] = 1.0

                # Always allow depot (to return and start new route)
                if len(visited_customers) < n:
                    mask[0, 0] = 1.0

                # Get logits
                logits = model.decoder(embeddings, context)
                logits = logits.squeeze(0)

                # Apply mask
                logits[mask.squeeze(0) == 0] = float('-inf')

                # Greedy action
                action = logits.argmax().item()

                # Update state
                total_cost += inst.distance(current_node, action)
                route.append(action)

                if action == 0:
                    # Returned to depot
                    remaining_cap = inst.capacity
                else:
                    visited_customers.add(action)
                    visited_set.add(action)
                    remaining_cap -= inst.demands[action - 1]

                current_node = action

            # Final return to depot
            if current_node != 0:
                total_cost += inst.distance(current_node, 0)
                route.append(0)

            costs.append(total_cost)
            routes.append(route)

    return costs, routes


# ============================================================================
# SECTION E: COMPARISON AND REPORTING
# ============================================================================

def run_comparison(instances, model=None, device="cpu"):
    """Run nearest-neighbor and model on the same instances, compare."""

    print(f"\n{'='*60}")
    print(f"Sanity Check: {len(instances)} CVRP instances, "
          f"n={instances[0].num_customers}")
    print(f"{'='*60}")

    # Nearest-neighbor baseline
    print("\nRunning nearest-neighbor heuristic...")
    t0 = time.time()
    nn_costs = []
    for inst in instances:
        cost, _ = nearest_neighbor_cvrp(inst)
        nn_costs.append(cost)
    nn_time = time.time() - t0
    nn_costs = np.array(nn_costs)

    print(f"  Mean cost:  {nn_costs.mean():.4f}")
    print(f"  Std:        {nn_costs.std():.4f}")
    print(f"  Min:        {nn_costs.min():.4f}")
    print(f"  Max:        {nn_costs.max():.4f}")
    print(f"  Total time: {nn_time:.2f}s ({nn_time/len(instances)*1000:.2f}ms/instance)")

    if model is not None:
        # Model inference
        print("\nRunning trained model (greedy)...")
        t0 = time.time()
        model_costs, model_routes = solve_with_model(model, instances, device)
        model_time = time.time() - t0
        model_costs = np.array(model_costs)

        print(f"  Mean cost:  {model_costs.mean():.4f}")
        print(f"  Std:        {model_costs.std():.4f}")
        print(f"  Min:        {model_costs.min():.4f}")
        print(f"  Max:        {model_costs.max():.4f}")
        print(f"  Total time: {model_time:.2f}s ({model_time/len(instances)*1000:.2f}ms/instance)")

        # Comparison
        improvement = (nn_costs.mean() - model_costs.mean()) / nn_costs.mean() * 100
        win_rate = (model_costs < nn_costs).mean() * 100

        print(f"\n{'='*60}")
        print(f"COMPARISON")
        print(f"{'='*60}")
        print(f"  Model vs NN improvement: {improvement:+.2f}%")
        print(f"  Model wins on {win_rate:.1f}% of instances")
        print()

        if improvement > 5:
            print("  VERDICT: Model is significantly better than NN.")
            print("  The model has learned useful routing strategies.")
        elif improvement > 0:
            print("  VERDICT: Model is slightly better than NN.")
            print("  Consider training longer for stronger results.")
        elif improvement > -5:
            print("  VERDICT: Model is roughly on par with NN.")
            print("  The model likely needs more training.")
        else:
            print("  VERDICT: Model is WORSE than NN.")
            print("  Something may be wrong. Check:")
            print("    1. Feature construction matches training code")
            print("    2. Context building matches training code")
            print("    3. Masking logic is correct")
            print("    4. Model has trained for enough episodes/epochs")

        # Per-instance comparison (first 10)
        print(f"\n{'='*60}")
        print(f"First 10 instances (detailed)")
        print(f"{'='*60}")
        print(f"{'Instance':>10} {'NN cost':>10} {'Model cost':>12} {'Diff':>10} {'Winner':>8}")
        print(f"{'-'*52}")
        for i in range(min(10, len(instances))):
            diff = model_costs[i] - nn_costs[i]
            winner = "Model" if model_costs[i] < nn_costs[i] else "NN"
            print(f"{i:>10} {nn_costs[i]:>10.4f} {model_costs[i]:>12.4f} "
                  f"{diff:>+10.4f} {winner:>8}")

    else:
        print("\n  No model loaded — showing NN baseline only.")
        print("  Fix model loading (Section A) to run comparison.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check: model vs nearest-neighbor")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--variant", type=str, default="cvrp",
                        choices=["cvrp", "cvrptw", "evrp", "evrptw"])
    parser.add_argument("--size", type=int, default=20,
                        help="Number of customers")
    parser.add_argument("--num-instances", type=int, default=100,
                        help="Number of test instances")
    parser.add_argument("--capacity", type=float, default=50.0,
                        help="Vehicle capacity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nn-only", action="store_true",
                        help="Run only nearest-neighbor (skip model)")
    args = parser.parse_args()

    # Generate instances
    print(f"Generating {args.num_instances} CVRP instances (n={args.size})...")
    instances = generate_cvrp_instances(
        args.num_instances, args.size, args.capacity, args.seed
    )

    # Load model
    model = None
    if args.checkpoint and not args.nn_only:
        try:
            model = load_your_model(args.checkpoint, args.device)
        except Exception as e:
            print(f"\nFailed to load model: {e}")
            print("Running NN-only comparison.")
            print("If the architecture doesn't match, adapt Section A.\n")

    # Run comparison
    run_comparison(instances, model, args.device)
