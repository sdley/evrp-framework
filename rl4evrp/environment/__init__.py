"""
EVRP Environment: Instance generation and environment dynamics.
"""

import numpy as np
import torch
from typing import Dict, Tuple


def generate_instance(n_customers: int = 15, seed: int = None,
                      charger_prob: float = 0.15,
                      cargo_cap: float = 30.0,
                      battery_cap: float = 100.0) -> dict:
    """
    Generate a random EVRP instance.
    
    Args:
        n_customers: Number of customers (excluding depot)
        seed: Random seed for reproducibility
        charger_prob: Fraction of nodes that become chargers
        cargo_cap: Vehicle cargo capacity
        battery_cap: Vehicle battery capacity
    
    Returns:
        Instance dict with coordinates, demands, node types, etc.
    """
    rng = np.random.RandomState(seed)
    n_total = n_customers + 1  # +1 depot at index 0
    
    # Generate node positions uniformly in [0, 1]^2
    coords = rng.uniform(0, 1, (n_total, 2)).astype(np.float32)
    
    # Generate demands for customers
    demands = np.zeros(n_total, dtype=np.float32)
    demands[1:] = rng.randint(1, 8, n_customers).astype(np.float32)
    
    # Determine node types: 0=depot, 1=customer, 2=charger
    node_types = np.zeros(n_total, dtype=int)
    node_types[1:] = 1  # Initially all customers
    
    chargers = rng.rand(n_customers) < charger_prob
    node_types[1:][chargers] = 2  # Some become chargers
    demands[node_types == 2] = 0  # Chargers have no demand
    
    return dict(
        coords=coords,
        demands=demands,
        node_types=node_types,
        cargo_cap=cargo_cap,
        battery_cap=battery_cap,
        n_nodes=n_total
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
    
    # Coordinates
    f[:, 0] = inst['coords'][:, 0]
    f[:, 1] = inst['coords'][:, 1]
    
    # Normalized demand
    f[:, 2] = inst['demands'] / inst['cargo_cap']
    
    # Node type indicators
    f[:, 3] = (inst['node_types'] == 2).astype(float)  # is_charger
    f[:, 4] = (inst['node_types'] == 0).astype(float)  # is_depot
    
    # Capacity normalizations
    f[:, 5] = inst['cargo_cap'] / 50.0
    f[:, 6] = inst['battery_cap'] / 200.0
    
    return torch.FloatTensor(f)


def make_dataset(n: int, n_customers: int = 15, seed0: int = 0,
                 cargo_cap: float = 30.0,
                 battery_cap: float = 100.0,
                 charger_prob: float = 0.15) -> list:
    """
    Generate multiple instances for training/evaluation.
    
    Args:
        n: Number of instances
        n_customers: Customers per instance
        seed0: Base seed
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


class EVRPEnv:
    """EVRP environment with proper battery recharge at depot."""
    
    def __init__(self, inst: dict, reward_mode: str = 'distance'):
        """
        Initialize EVRP environment.
        
        Args:
            inst: Instance dict from generate_instance()
            reward_mode: 'distance' (minimize) or 'inverse_distance' (reward short hops)
        """
        self.inst = inst
        self.reward_mode = reward_mode
        self.n = inst['n_nodes']
        self.coords = inst['coords']
        self.demands = inst['demands'].copy()
        self.types = inst['node_types']
        self.cargo_cap = inst['cargo_cap']
        self.battery_cap = inst['battery_cap']
        
        # Precompute distance matrix
        self.D = self._build_dist()
        self.d_max = self.D.max() + 1e-9
        
        # State tracking
        self.cur = 0
        self.battery = self.battery_cap
        self.cargo = self.cargo_cap
        self.visited = np.zeros(self.n, dtype=bool)
        self.done = False
        self.route = []
        self.total_d = 0.0
        self.steps = 0
        
        # Diagnostic logs
        self.battery_log = []
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.masked_counts = []
    
    def _build_dist(self) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        c = self.coords
        diff = c[:, None] - c[None, :]
        return np.sqrt((diff**2).sum(-1)).astype(np.float32)
    
    def reset(self) -> dict:
        """Reset environment to initial state."""
        self.cur = 0
        self.battery = self.battery_cap
        self.cargo = self.cargo_cap
        self.visited = np.zeros(self.n, dtype=bool)
        self.visited[0] = True
        self.done = False
        self.route = [0]
        self.total_d = 0.0
        self.steps = 0
        
        # Reset logs
        self.battery_log = [self.battery]
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.masked_counts = []
        
        return self._obs()
    
    def _obs(self) -> dict:
        """Get current observation."""
        return dict(
            node_features=build_node_features(self.inst),
            current_node=self.cur,
            battery_norm=np.float32(self.battery / self.battery_cap),
            cargo_norm=np.float32(self.cargo / self.cargo_cap),
            visited_mask=self.visited.copy(),
            valid_mask=self._valid_mask(),
        )
    
    def _valid_mask(self) -> np.ndarray:
        """
        Compute valid action mask.
        
        True = can visit this node.
        Valid nodes:
        - Chargers: reachable
        - Depot: always reachable
        - Customers: unvisited, have cargo, can reach and return to depot
        """
        valid = np.zeros(self.n, dtype=bool)
        
        for j in range(self.n):
            if j == self.cur:
                continue
            
            d_to_j = self.D[self.cur, j]
            
            # Battery constraint
            if self.battery < d_to_j:
                continue
            
            # Charger: always ok if reachable
            if self.types[j] == 2:
                valid[j] = True
                continue
            
            # Depot: always ok if reachable
            if j == 0:
                valid[j] = True
                continue
            
            # Customer: need to check cargo and return feasibility
            d_back = self.D[j, 0]
            if (not self.visited[j]) and self.cargo >= self.demands[j]:
                if self.battery >= d_to_j + d_back:
                    valid[j] = True
        
        # If no valid actions, force depot
        if not valid.any():
            valid[0] = True
        
        self.masked_counts.append(int((~valid).sum()))
        return valid
    
    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        Take one step in the environment.
        
        Args:
            action: Next node to visit
        
        Returns:
            (next_obs, reward, done, info)
        """
        d = self.D[self.cur, action]
        self.battery -= d
        self.total_d += d
        self.steps += 1
        
        # Check battery violation
        batt_viol = self.battery < -1e-3
        if batt_viol:
            self.batt_violations += 1
        
        # Compute base reward
        if self.reward_mode == 'distance':
            reward = -float(d) / self.d_max
        else:  # inverse_distance
            reward = 0.4 / (float(d) / self.d_max + 0.05)
        
        served_now = False
        
        # Process node visit
        if self.types[action] == 2:  # Charger
            self.battery = self.battery_cap
            self.charger_visits += 1
            reward -= 0.05
        elif action == 0:  # Depot — recharge both battery and cargo
            self.battery = self.battery_cap
            self.cargo = self.cargo_cap
            if not self.visited[1:].all():
                reward -= 0.3  # Early return penalty
        else:  # Customer
            cargo_viol = self.cargo < self.demands[action]
            if cargo_viol:
                self.cargo_violations += 1
            
            self.cargo -= self.demands[action]
            self.visited[action] = True
            served_now = True
            reward += 0.2  # Service bonus
        
        self.cur = action
        self.route.append(action)
        self.battery_log.append(max(0.0, self.battery))
        
        # Check completion
        customers = np.where(self.types == 1)[0]
        all_served = self.visited[customers].all()
        
        if all_served and action == 0:
            reward += 2.0  # Completion bonus
        
        if batt_viol:
            reward -= 1.0
        
        # Termination condition
        self.done = (all_served and action == 0) or self.steps >= 4 * self.n
        
        info = dict(
            dist=float(d),
            total_dist=self.total_d,
            all_served=bool(all_served),
            n_customers_served=int(self.visited[customers].sum()),
            battery=max(0.0, self.battery),
            cargo=self.cargo,
            charger_visits=self.charger_visits,
            batt_violations=self.batt_violations,
            cargo_violations=self.cargo_violations,
            node_type=int(self.types[action]),
        )
        
        return self._obs(), reward, self.done, info
