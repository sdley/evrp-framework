"""Train an A2C agent on the EVRP and save checkpoints."""

import argparse
import json
import sys
from pathlib import Path

# Allow `uv run python scripts/train.py` from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rl4evrp.agents import A2CAgent
from rl4evrp.utils import OnTheFlyInstancePool, train_agent, evaluate_agent
from rl4evrp.environment import generate_instance


def parse_args():
    p = argparse.ArgumentParser(description="Train A2C on EVRP")

    # Problem
    p.add_argument("--n-customers", type=int, default=15,
                   help="Customers per instance (default: 15)")
    p.add_argument("--charger-prob", type=float, default=0.15,
                   help="Charger node probability (default: 0.15)")
    p.add_argument("--cargo-cap", type=float, default=30.0)
    p.add_argument("--battery-cap", type=float, default=100.0)

    # Training
    p.add_argument("--n-episodes", type=int, default=500,
                   help="Training episodes (default: 500)")
    p.add_argument("--pool-size", type=int, default=1000,
                   help="On-the-fly instance pool size (default: 1000)")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--enc-type", choices=["gat", "mlp"], default="gat")

    # Output
    p.add_argument("--out-dir", type=Path, default=Path("results/train"),
                   help="Checkpoint and results directory")
    p.add_argument("--save-interval", type=int, default=100,
                   help="Save checkpoint every N episodes (default: 100)")
    p.add_argument("--n-eval", type=int, default=20,
                   help="Evaluation instances (default: 20)")
    p.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', or 'auto' (default: auto)")

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")

    def gen(seed):
        return generate_instance(
            n_customers=args.n_customers,
            seed=seed,
            charger_prob=args.charger_prob,
            cargo_cap=args.cargo_cap,
            battery_cap=args.battery_cap,
        )

    train_pool = OnTheFlyInstancePool(gen, size=args.pool_size, seed_offset=0)
    eval_instances = [gen(seed=10_000 + i) for i in range(args.n_eval)]

    agent = A2CAgent(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        gamma=args.gamma,
        enc_type=args.enc_type,
        n_episodes=args.n_episodes,
        device=device,
    )
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"params : {n_params:,}")
    print(f"episodes: {args.n_episodes}")
    print(f"out_dir : {args.out_dir}")
    print()

    results = train_agent(
        agent,
        train_instances=train_pool,
        n_episodes=args.n_episodes,
        device=device,
        save_dir=args.out_dir / "checkpoints",
        eval_instances=eval_instances,
        save_interval=args.save_interval,
    )

    # Final evaluation
    print("\nFinal evaluation ...")
    stats = evaluate_agent(agent, eval_instances, device=device, greedy=True)
    print(f"mean reward  : {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
    print(f"mean distance: {stats['mean_distance']:.4f} ± {stats['std_distance']:.4f}")

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.json"
    with open(results_path, "w") as f:
        payload = {
            "train_rewards": results["train_rewards"],
            "losses": results["losses"],
            "eval_rewards": results["eval_rewards"] or [],
            "final_eval": {
                "mean_reward": stats["mean_reward"],
                "std_reward": stats["std_reward"],
                "mean_distance": stats["mean_distance"],
                "std_distance": stats["std_distance"],
            },
        }
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save final model
    model_path = args.out_dir / "agent_final.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to  {model_path}")

    # Save architecture metadata so evaluate.py can validate args
    arch_path = args.out_dir / "arch.json"
    with open(arch_path, "w") as f:
        json.dump({
            "embed_dim": args.embed_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "enc_type": args.enc_type,
        }, f, indent=2)
    print(f"Arch config saved to {arch_path}")


if __name__ == "__main__":
    main()
