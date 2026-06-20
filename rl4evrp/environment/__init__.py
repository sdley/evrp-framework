"""
EVRP Environment: Instance generation and environment dynamics.
"""

import numpy as np
import torch
from typing import Dict, Tuple


def generate_instance(n_customers: int = 15, seed: int = None,
                      charger_prob: float = 0.15,
                      cargo_cap: float = 30.0,
                      battery_cap: float = 100.0,
                      tw_enabled: bool = True,
                      time_horizon: float = 4.0,
                      tw_width_min: float = 0.4,
                      tw_width_max: float = 1.2,
                      service_time_customer: float = 0.05,
                      service_time_charger: float = 0.03,
                      service_time_depot: float = 0.0,
                      reward_params: Dict = None) -> dict:
    """
    Generate a random EVRP instance.
    
    Args:
        n_customers: Number of customers (excluding depot)
        seed: Random seed for reproducibility
        charger_prob: Fraction of nodes that become chargers
        cargo_cap: Vehicle cargo capacity
        battery_cap: Vehicle battery capacity
    
    Returns:
        Instance dict with coordinates, demands, node types, and optional time windows.
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
    
    # Time-window fields
    ready_times = np.zeros(n_total, dtype=np.float32)
    due_times = np.full(n_total, float(time_horizon), dtype=np.float32)
    service_times = np.zeros(n_total, dtype=np.float32)

    if tw_enabled:
        customer_idx = np.where(node_types == 1)[0]
        n_customer_nodes = len(customer_idx)
        if n_customer_nodes > 0:
            earliest = rng.uniform(0.0, 0.65 * time_horizon, size=n_customer_nodes)
            widths = rng.uniform(tw_width_min, tw_width_max, size=n_customer_nodes)
            latest = np.minimum(earliest + widths, time_horizon)
            latest = np.maximum(latest, earliest + 1e-3)

            ready_times[customer_idx] = earliest.astype(np.float32)
            due_times[customer_idx] = latest.astype(np.float32)

        service_times[node_types == 1] = np.float32(service_time_customer)
        service_times[node_types == 2] = np.float32(service_time_charger)
        service_times[0] = np.float32(service_time_depot)

        # Depot: not a customer TW, open over whole horizon
        ready_times[0] = 0.0
        due_times[0] = float(time_horizon)

        # Chargers: not customer TWs either, open over whole horizon
        charger_idx = np.where(node_types == 2)[0]
        ready_times[charger_idx] = 0.0
        due_times[charger_idx] = float(time_horizon)

    if reward_params is None:
        reward_params = {
            'service_bonus': 0.2,
            'completion_bonus': 2.0,
            'charger_penalty': 0.05,
            'early_return_penalty': 0.3,
            'battery_violation_penalty': 1.0,
            'time_window_violation_penalty': 1.5,
            'waiting_time_penalty': 0.05,
        }

    return dict(
        coords=coords,
        demands=demands,
        node_types=node_types,
        cargo_cap=cargo_cap,
        battery_cap=battery_cap,
        n_nodes=n_total,
        tw_enabled=bool(tw_enabled),
        time_horizon=float(time_horizon),
        ready_times=ready_times,
        due_times=due_times,
        service_times=service_times,
        reward_params=dict(reward_params)
    )


def build_node_features(inst: dict) -> torch.Tensor:
    """
    Build node feature tensor from instance.

    Default features:
    [x, y, demand_norm, is_charger, is_depot,
     cargo_cap_norm, battery_cap_norm,
     tw_ready_norm, tw_due_norm, service_time_norm]

    Supports alternative state definitions through:
      inst["state_feature_names"] = list of feature names

    Also supports old feature-mask ablations:
      inst["feature_mask"] = binary mask of length feature_dim
    """
    n = inst["n_nodes"]

    DEFAULT_FEATURES = [
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
    ]

    feature_names = inst.get("state_feature_names", DEFAULT_FEATURES)
    feature_dim = len(feature_names)

    # Base normalized quantities
    cargo_cap = float(inst["cargo_cap"])
    battery_cap = float(inst["battery_cap"])
    time_horizon = float(inst.get("time_horizon", 1.0))
    time_horizon = max(time_horizon, 1e-6)

    coords = np.asarray(inst["coords"], dtype=np.float32)
    demands = np.asarray(inst["demands"], dtype=np.float32)
    node_types = np.asarray(inst["node_types"], dtype=np.int32)

    ready_times = np.asarray(
        inst.get("ready_times", np.zeros(n, dtype=np.float32)),
        dtype=np.float32
    )
    due_times = np.asarray(
        inst.get("due_times", np.full(n, time_horizon, dtype=np.float32)),
        dtype=np.float32
    )
    service_times = np.asarray(
        inst.get("service_times", np.zeros(n, dtype=np.float32)),
        dtype=np.float32
    )

    # Precompute reusable feature arrays
    feature_bank = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "demand_normalized": demands / max(cargo_cap, 1e-6),
        "is_charger": (node_types == 2).astype(np.float32),
        "is_depot": (node_types == 0).astype(np.float32),
        "cargo_capacity_norm": np.full(n, cargo_cap / 50.0, dtype=np.float32),
        "battery_capacity_norm": np.full(n, battery_cap / 200.0, dtype=np.float32),
        "tw_ready_norm": ready_times / time_horizon,
        "tw_due_norm": due_times / time_horizon,
        "service_time_norm": service_times / time_horizon,

        # Extra engineered features for state-formulation study
        "demand_over_cargo": demands / max(cargo_cap, 1e-6),
        "tw_window_width_norm": np.maximum(0.0, due_times - ready_times) / time_horizon,

        # Dummy channel
        "zeros": np.zeros(n, dtype=np.float32),
    }

    # Build features dynamically from requested feature names
    f = np.zeros((n, feature_dim), dtype=np.float32)
    for k, name in enumerate(feature_names):
        if name not in feature_bank:
            raise ValueError(f"Unknown state feature name: {name}")
        f[:, k] = feature_bank[name]

    # Optional old feature-mask support
    feature_mask = inst.get("feature_mask", None)
    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=np.float32)
        if feature_mask.shape[0] != f.shape[1]:
            raise ValueError(
                f"feature_mask has length {feature_mask.shape[0]}, "
                f"but node features have dimension {f.shape[1]}"
            )
        f = f * feature_mask[None, :]

    return torch.FloatTensor(f)


def make_dataset(n: int, n_customers: int = 15, seed0: int = 0,
                 cargo_cap: float = 30.0,
                 battery_cap: float = 100.0,
                 charger_prob: float = 0.15,
                 tw_enabled: bool = True,
                 time_horizon: float = 4.0,
                 tw_width_min: float = 0.4,
                 tw_width_max: float = 1.2,
                 service_time_customer: float = 0.05,
                 service_time_charger: float = 0.03,
                 service_time_depot: float = 0.0,
                 reward_params: Dict = None) -> list:
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
                         battery_cap=battery_cap,
                         tw_enabled=tw_enabled,
                         time_horizon=time_horizon,
                         tw_width_min=tw_width_min,
                         tw_width_max=tw_width_max,
                         service_time_customer=service_time_customer,
                         service_time_charger=service_time_charger,
                         service_time_depot=service_time_depot,
                         reward_params=reward_params)
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

        # EVRPTW-specific state
        self.tw_enabled = bool(inst.get('tw_enabled', False))
        self.time_horizon = float(inst.get('time_horizon', 1.0))
        self.ready_times = np.asarray(
            inst.get('ready_times', np.zeros(self.n, dtype=np.float32)),
            dtype=np.float32,
        )
        self.due_times = np.asarray(
            inst.get('due_times', np.full(self.n, self.time_horizon, dtype=np.float32)),
            dtype=np.float32,
        )
        self.service_times = np.asarray(
            inst.get('service_times', np.zeros(self.n, dtype=np.float32)),
            dtype=np.float32,
        )
        self.reward_params = dict(inst.get('reward_params', {}))
        
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
        self.time = 0.0
        self.steps = 0
        
        # Diagnostic logs
        self.battery_log = []
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.tw_violations = 0
        self.masked_counts = []
        self.wait_time_log = []
    
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
        self.time = 0.0
        self.steps = 0
        
        # Reset logs
        self.battery_log = [self.battery]
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.tw_violations = 0
        self.masked_counts = []
        self.wait_time_log = [0.0]
        
        return self._obs()
    
    def _obs(self) -> dict:
        """Get current observation."""
        return dict(
            node_features=build_node_features(self.inst),
            current_node=self.cur,
            battery_norm=np.float32(self.battery / self.battery_cap),
            cargo_norm=np.float32(self.cargo / self.cargo_cap),
            time_norm=np.float32(self.time / max(self.time_horizon, 1e-6)),
            current_time=np.float32(self.time),
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
            energy_rate = 20.0
            if self.battery < energy_rate * d_to_j:
                continue

            # Time-window feasibility: arrival must be before due time.
            if self.tw_enabled:
                arrival = self.time + float(d_to_j)
                if arrival > float(self.due_times[j]) + 1e-9:
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
                if self.battery >= energy_rate * (d_to_j + d_back):
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
        #self.battery -= d
        energy_rate = 20.0
        self.battery -= energy_rate * d
        self.total_d += d
        self.steps += 1

        # Time progression with optional waiting and service.
        arrival_time = self.time + float(d)
        wait_time = 0.0
        if self.tw_enabled:
            wait_time = max(0.0, float(self.ready_times[action] - arrival_time))
        start_service_time = arrival_time + wait_time
        service_time = float(self.service_times[action]) if self.tw_enabled else 0.0

        tw_viol = False
        if self.tw_enabled and start_service_time > float(self.due_times[action]) + 1e-9:
            tw_viol = True
            self.tw_violations += 1

        self.time = start_service_time + service_time
        self.wait_time_log.append(wait_time)
        
        # Check battery violation
        batt_viol = self.battery < -1e-3
        if batt_viol:
            self.batt_violations += 1
        
        # Compute base reward
        if self.reward_mode == 'distance':
            reward = -float(d) / self.d_max
        else:  # inverse_distance
            reward = 0.4 / (float(d) / self.d_max + 0.05)

        service_bonus = float(self.reward_params.get('service_bonus', 0.2))
        completion_bonus = float(self.reward_params.get('completion_bonus', 2.0))
        charger_penalty = float(self.reward_params.get('charger_penalty', 0.05))
        early_return_penalty = float(self.reward_params.get('early_return_penalty', 0.3))
        battery_violation_penalty = float(self.reward_params.get('battery_violation_penalty', 1.0))
        tw_violation_penalty = float(self.reward_params.get('time_window_violation_penalty', 1.5))
        waiting_time_penalty = float(self.reward_params.get('waiting_time_penalty', 0.05))

        if self.tw_enabled and self.time_horizon > 0.0:
            reward -= waiting_time_penalty * (wait_time / self.time_horizon)
        
        served_now = False
        
        # Process node visit
        if self.types[action] == 2:  # Charger
            self.battery = self.battery_cap
            self.charger_visits += 1
            reward -= charger_penalty
        elif action == 0:  # Depot — recharge both battery and cargo
            self.battery = self.battery_cap
            self.cargo = self.cargo_cap
            if not self.visited[1:].all():
                reward -= early_return_penalty
        else:  # Customer
            cargo_viol = self.cargo < self.demands[action]
            if cargo_viol:
                self.cargo_violations += 1
            
            self.cargo -= self.demands[action]
            self.visited[action] = True
            served_now = True
            reward += service_bonus
        
        self.cur = action
        self.route.append(action)
        self.battery_log.append(max(0.0, self.battery))
        
        # Check completion
        customers = np.where(self.types == 1)[0]
        all_served = self.visited[customers].all()
        
        if all_served and action == 0:
            reward += completion_bonus
        
        if batt_viol:
            reward -= battery_violation_penalty

        if tw_viol:
            reward -= tw_violation_penalty
        
        # Termination condition
        self.done = (all_served and action == 0) or self.steps >= 4 * self.n
        
        info = dict(
            dist=float(d),
            total_dist=self.total_d,
            all_served=bool(all_served),
            n_customers_served=int(self.visited[customers].sum()),
            battery=max(0.0, self.battery),
            cargo=self.cargo,
            current_time=self.time,
            wait_time=wait_time,
            charger_visits=self.charger_visits,
            batt_violations=self.batt_violations,
            cargo_violations=self.cargo_violations,
            tw_violations=self.tw_violations,
            node_type=int(self.types[action]),
        )
        
        return self._obs(), reward, self.done, info