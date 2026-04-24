import numpy as np
from typing import Tuple

from .instances import build_node_features


class EVRPEnv:
    """EVRP environment with battery recharge at depot and charger nodes."""

    def __init__(self, inst: dict, reward_mode: str = 'distance', max_steps: int = None):
        """
        Args:
            inst: Instance dict from generate_instance()
            reward_mode: 'distance' (minimize) or 'inverse_distance'
            max_steps: Hard step limit per episode. Defaults to 4 * n_nodes.
        """
        self.inst = inst
        self.reward_mode = reward_mode
        self.n = inst['n_nodes']
        self.max_steps = max_steps if max_steps is not None else 4 * self.n
        self.coords = inst['coords']
        self.demands = inst['demands'].copy()
        self.types = inst['node_types']
        self.cargo_cap = inst['cargo_cap']
        self.battery_cap = inst['battery_cap']

        self.D = self._build_dist()
        self.d_max = self.D.max() + 1e-9

        self.cur = 0
        self.battery = self.battery_cap
        self.cargo = self.cargo_cap
        self.visited = np.zeros(self.n, dtype=bool)
        self.done = False
        self.route = []
        self.total_d = 0.0
        self.steps = 0

        self.battery_log = []
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.masked_counts = []

    def _build_dist(self) -> np.ndarray:
        c = self.coords
        diff = c[:, None] - c[None, :]
        return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

    def reset(self) -> dict:
        self.cur = 0
        self.battery = self.battery_cap
        self.cargo = self.cargo_cap
        self.visited = np.zeros(self.n, dtype=bool)
        self.visited[0] = True
        self.done = False
        self.route = [0]
        self.total_d = 0.0
        self.steps = 0

        self.battery_log = [self.battery]
        self.charger_visits = 0
        self.batt_violations = 0
        self.cargo_violations = 0
        self.masked_counts = []

        return self._obs()

    def _obs(self) -> dict:
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
        True = node is a valid next action.

        Rules:
        - Chargers: reachable by battery
        - Depot: always reachable
        - Customers: unvisited, sufficient cargo, reachable and can return to depot
        """
        valid = np.zeros(self.n, dtype=bool)

        for j in range(self.n):
            if j == self.cur:
                continue
            d_to_j = self.D[self.cur, j]
            if self.battery < d_to_j:
                continue
            if self.types[j] == 2:
                valid[j] = True
                continue
            if j == 0:
                valid[j] = True
                continue
            d_back = self.D[j, 0]
            if (not self.visited[j]) and self.cargo >= self.demands[j]:
                if self.battery >= d_to_j + d_back:
                    valid[j] = True

        if not valid.any():
            valid[0] = True

        self.masked_counts.append(int((~valid).sum()))
        return valid

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        Args:
            action: Next node index to visit
        Returns:
            (next_obs, reward, done, info)
        """
        d = self.D[self.cur, action]
        self.battery -= d
        self.total_d += d
        self.steps += 1

        batt_viol = self.battery < -1e-3
        if batt_viol:
            self.batt_violations += 1

        if self.reward_mode == 'distance':
            reward = -float(d) / self.d_max
        else:
            reward = 0.4 / (float(d) / self.d_max + 0.05)

        if self.types[action] == 2:
            self.battery = self.battery_cap
            self.charger_visits += 1
            reward -= 0.05
        elif action == 0:
            self.battery = self.battery_cap
            self.cargo = self.cargo_cap
            if not self.visited[1:].all():
                reward -= 0.3
        else:
            cargo_viol = self.cargo < self.demands[action]
            if cargo_viol:
                self.cargo_violations += 1
            self.cargo -= self.demands[action]
            self.visited[action] = True
            reward += 0.2

        self.cur = action
        self.route.append(action)
        self.battery_log.append(max(0.0, self.battery))

        customers = np.where(self.types == 1)[0]
        all_served = self.visited[customers].all()

        if all_served and action == 0:
            reward += 2.0
        if batt_viol:
            reward -= 1.0

        self.done = (all_served and action == 0) or self.steps >= self.max_steps

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
