import numpy as np
import torch


def generate_instance(n_customers: int = 15, seed: int = None,
                      charger_prob: float = 0.15,
                      cargo_cap: float = 30.0,
                      battery_cap: float = 100.0) -> dict:
    """
    Generate a random EVRP instance.

    Args:
        n_customers: Number of customers (excluding depot)
        seed: Random seed for reproducibility
        charger_prob: Fraction of customer nodes that become chargers
        cargo_cap: Vehicle cargo capacity
        battery_cap: Vehicle battery capacity

    Returns:
        Instance dict with coords, demands, node_types, capacities, n_nodes.
    """
    rng = np.random.RandomState(seed)
    n_total = n_customers + 1  # index 0 is depot

    coords = rng.uniform(0, 1, (n_total, 2)).astype(np.float32)

    demands = np.zeros(n_total, dtype=np.float32)
    demands[1:] = rng.randint(1, 8, n_customers).astype(np.float32)

    node_types = np.zeros(n_total, dtype=int)
    node_types[1:] = 1
    chargers = rng.rand(n_customers) < charger_prob
    node_types[1:][chargers] = 2
    demands[node_types == 2] = 0

    return dict(
        coords=coords,
        demands=demands,
        node_types=node_types,
        cargo_cap=cargo_cap,
        battery_cap=battery_cap,
        n_nodes=n_total,
    )


def build_node_features(inst: dict) -> torch.Tensor:
    """
    Build node feature tensor from instance.

    Features: [x, y, demand_norm, is_charger, is_depot, cargo_cap_norm, battery_cap_norm]

    Args:
        inst: Instance dict from generate_instance()

    Returns:
        Tensor of shape (n_nodes, 7)
    """
    n = inst['n_nodes']
    f = np.zeros((n, 7), dtype=np.float32)
    f[:, 0] = inst['coords'][:, 0]
    f[:, 1] = inst['coords'][:, 1]
    f[:, 2] = inst['demands'] / inst['cargo_cap']
    f[:, 3] = (inst['node_types'] == 2).astype(float)
    f[:, 4] = (inst['node_types'] == 0).astype(float)
    f[:, 5] = inst['cargo_cap'] / 50.0
    f[:, 6] = inst['battery_cap'] / 200.0
    return torch.FloatTensor(f)


def make_dataset(n: int, n_customers: int = 15, seed0: int = 0,
                 cargo_cap: float = 30.0,
                 battery_cap: float = 100.0,
                 charger_prob: float = 0.15) -> list:
    """
    Generate multiple EVRP instances for training/evaluation.

    Args:
        n: Number of instances
        n_customers: Customers per instance
        seed0: Base seed (instance i uses seed0 + i)
        cargo_cap: Vehicle cargo capacity
        battery_cap: Vehicle battery capacity
        charger_prob: Charger probability

    Returns:
        List of instance dicts
    """
    return [
        generate_instance(n_customers, seed=seed0 + i,
                          charger_prob=charger_prob,
                          cargo_cap=cargo_cap,
                          battery_cap=battery_cap)
        for i in range(n)
    ]
