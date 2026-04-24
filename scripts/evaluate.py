"""Evaluate a saved A2C checkpoint on fresh EVRP instances."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rl4evrp.agents import A2CAgent
from rl4evrp.utils import evaluate_agent
from rl4evrp.environment import generate_instance


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved rl4evrp checkpoint")

    p.add_argument("checkpoint", type=Path,
                   help="Path to .pt checkpoint file")

    # Problem (must match training settings)
    p.add_argument("--n-customers", type=int, default=15)
    p.add_argument("--charger-prob", type=float, default=0.15)
    p.add_argument("--cargo-cap", type=float, default=30.0)
    p.add_argument("--battery-cap", type=float, default=100.0)

    # Model (must match training settings)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--enc-type", choices=["gat", "mlp"], default="gat")

    # Evaluation
    p.add_argument("--n-eval", type=int, default=100,
                   help="Number of evaluation instances (default: 100)")
    p.add_argument("--seed", type=int, default=99999,
                   help="Base seed for evaluation instances (default: 99999)")
    p.add_argument("--greedy", action="store_true", default=True)
    p.add_argument("--stochastic", dest="greedy", action="store_false",
                   help="Use stochastic policy instead of greedy")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save evaluation results as JSON")

    return p.parse_args()


def _validate_arch(checkpoint: Path, args) -> None:
    """Warn and exit if saved arch.json disagrees with CLI args."""
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        arch_file = parent / "arch.json"
        if arch_file.exists():
            with open(arch_file) as f:
                saved = json.load(f)
            mismatches = {
                k: (saved[k], getattr(args, k))
                for k in saved
                if saved[k] != getattr(args, k, None)
            }
            if mismatches:
                print("ERROR: Architecture mismatch between checkpoint and --args:")
                for k, (saved_v, given_v) in mismatches.items():
                    print(f"  --{k.replace('_', '-')}: checkpoint={saved_v!r}, given={given_v!r}")
                print("Re-run with the correct flags or omit them to use the saved values.")
                sys.exit(1)
            return  # found and validated


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    _validate_arch(args.checkpoint, args)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild agent architecture and load weights
    agent = A2CAgent(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        enc_type=args.enc_type,
        n_episodes=1,          # irrelevant for evaluation
        device=device,
    )
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"Loaded: {args.checkpoint}")

    instances = [
        generate_instance(
            n_customers=args.n_customers,
            seed=args.seed + i,
            charger_prob=args.charger_prob,
            cargo_cap=args.cargo_cap,
            battery_cap=args.battery_cap,
        )
        for i in range(args.n_eval)
    ]

    policy = "greedy" if args.greedy else "stochastic"
    print(f"Evaluating {args.n_eval} instances ({policy}) on {device} ...")

    stats = evaluate_agent(agent, instances, device=device, greedy=args.greedy)

    print()
    print(f"mean reward  : {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
    print(f"mean distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "n_eval": args.n_eval,
                "policy": policy,
                "mean_reward": stats["mean_reward"],
                "std_reward": stats["std_reward"],
                "mean_distance": stats["mean_distance"],
                "std_distance": stats["std_distance"],
                "rewards": stats["rewards"],
                "distances": stats["distances"],
            }, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
