import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from rl4evrp.environment import EVRPEnv
from .pool import InstanceProvider, _resolve_training_instance


def run_episode(agent, inst: dict, device: str = 'cpu', greedy: bool = False,
                collect_traces: bool = False,
                bat_perturb: Optional[float] = None,
                cargo_perturb: Optional[float] = None) -> Tuple:
    """
    Run a single EVRP episode.

    Args:
        agent: A2CAgent instance
        inst: Instance dict from generate_instance()
        device: Torch device string
        greedy: Use greedy policy (no sampling)
        collect_traces: Collect per-step attention traces for XAI
        bat_perturb: Battery perturbation factor (counterfactual analysis)
        cargo_perturb: Cargo perturbation factor (counterfactual analysis)

    Returns:
        (total_reward, route, total_distance, final_info, transitions, traces, env)
    """
    env = EVRPEnv(inst, reward_mode=inst.get('reward_mode', 'distance'))
    obs = env.reset()
    done = False
    total_reward = 0.0
    transitions = []
    traces = [] if collect_traces else None
    use_grad = (not greedy) or collect_traces

    while not done:
        obs_use = obs
        if bat_perturb is not None or cargo_perturb is not None:
            obs_use = dict(obs)
            if bat_perturb is not None:
                obs_use['battery_norm'] = float(
                    np.clip(obs['battery_norm'] * bat_perturb, 0, 1)
                )
            if cargo_perturb is not None:
                obs_use['cargo_norm'] = float(
                    np.clip(obs['cargo_norm'] * cargo_perturb, 0, 1)
                )

        with torch.set_grad_enabled(use_grad):
            action, log_prob, entropy, value = agent.select_action(obs_use, greedy=greedy)

        if collect_traces:
            scores, _, _, dec_attn, enc_attn = agent._forward(obs_use, return_attn=True)
            probs = torch.softmax(scores.squeeze(0), -1)

            attn_avg = (
                dec_attn[0].mean(0).squeeze(0).detach().cpu().tolist()
                if dec_attn is not None else [0.0] * env.n
            )
            enc_attn_row = (
                enc_attn[0].mean(0)[obs['current_node']].detach().cpu().tolist()
                if enc_attn is not None else [0.0] * env.n
            )
            top3_p, top3_n = probs.topk(min(3, probs.shape[0]))

            traces.append(dict(
                step=len(traces),
                from_node=int(obs['current_node']),
                to_node=int(action),
                node_type=int(inst['node_types'][action]),
                battery_norm=float(obs['battery_norm']),
                cargo_norm=float(obs['cargo_norm']),
                battery_abs=float(obs['battery_norm'] * inst['battery_cap']),
                cargo_abs=float(obs['cargo_norm'] * inst['cargo_cap']),
                action_prob=float(probs[action].item()),
                action_logit=float(scores.squeeze(0)[action].item()),
                raw_logits=scores.squeeze(0).detach().cpu().tolist(),
                top3_nodes=top3_n.tolist(),
                top3_probs=top3_p.tolist(),
                valid_mask=obs['valid_mask'].tolist(),
                invalid_count=int((~obs['valid_mask']).sum()),
                dec_attn=attn_avg,
                enc_attn=enc_attn_row,
                dist_to_action=float(env.D[obs['current_node'], action]),
                dist_to_depot=float(env.D[action, 0]),
                is_charger=int(inst['node_types'][action] == 2),
                is_depot=int(action == 0),
                is_forced=int(np.sum(obs['valid_mask']) <= 1),
            ))

        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        transitions.append(dict(
            reward=reward,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        ))

        obs = next_obs

    return total_reward, env.route, env.total_d, info, transitions, traces, env


def train_agent(agent, train_instances: InstanceProvider, n_episodes: int = 100,
                device: str = 'cpu', save_dir: Optional[Path] = None,
                eval_instances: Optional[List[Dict]] = None,
                save_interval: int = 10) -> Dict:
    """
    Train A2C agent for a fixed number of episodes.

    Args:
        agent: A2CAgent instance
        train_instances: Sequence of instance dicts or callable f(episode) -> dict
        n_episodes: Number of training episodes
        device: Torch device string
        save_dir: Directory for checkpoints (None = no saving)
        eval_instances: Instances for periodic evaluation
        save_interval: Checkpoint / eval frequency

    Returns:
        Dict with train_rewards, eval_rewards, losses, entropies
    """
    agent.train()
    agent.to(device)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    train_rewards, losses, entropies = [], [], []
    eval_rewards = [] if eval_instances else None

    for episode in range(n_episodes):
        inst = _resolve_training_instance(train_instances, episode)
        inst['reward_mode'] = 'distance'

        with torch.enable_grad():
            episode_reward, route, total_dist, info, transitions, _, _ = run_episode(
                agent, inst, device=device, greedy=False
            )

        with torch.enable_grad():
            loss, ent = agent.update([
                {'r': t['reward'], 'lp': t['log_prob'],
                 'ent': t['entropy'], 'val': t['value']}
                for t in transitions
            ])

        train_rewards.append(episode_reward)
        losses.append(loss)
        entropies.append(ent)

        if eval_instances and (episode + 1) % save_interval == 0:
            eval_rews = []
            for eval_inst in eval_instances[:5]:
                eval_inst = dict(eval_inst)
                eval_inst['reward_mode'] = 'distance'
                eval_reward, _, _, _, _, _, _ = run_episode(
                    agent, eval_inst, device=device, greedy=True
                )
                eval_rews.append(eval_reward)
            avg_eval = np.mean(eval_rews)
            eval_rewards.append(avg_eval)
            print(f"Episode {episode+1}: train={episode_reward:.3f}, "
                  f"eval={avg_eval:.3f}, loss={loss:.4f}")
        elif (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: train={episode_reward:.3f}, loss={loss:.4f}")

        if save_dir and (episode + 1) % save_interval == 0:
            torch.save(agent.state_dict(), save_dir / f"agent_episode_{episode+1}.pt")

    return dict(
        train_rewards=train_rewards,
        eval_rewards=eval_rewards,
        losses=losses,
        entropies=entropies,
    )


def evaluate_agent(agent, instances: List[Dict], device: str = 'cpu',
                   greedy: bool = True, n_eval: Optional[int] = None) -> Dict:
    """
    Evaluate agent on a set of instances.

    Args:
        agent: A2CAgent instance
        instances: Evaluation instance dicts
        device: Torch device string
        greedy: Use greedy policy
        n_eval: Number of instances to use (None = all)

    Returns:
        Dict with mean/std reward and distance, plus raw lists
    """
    instances = instances[:n_eval] if n_eval else instances
    rewards, distances, routes = [], [], []

    for inst in instances:
        inst = dict(inst)
        inst['reward_mode'] = 'distance'
        reward, route, dist, info, _, _, _ = run_episode(
            agent, inst, device=device, greedy=greedy
        )
        rewards.append(reward)
        distances.append(dist)
        routes.append(route)

    return dict(
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        mean_distance=np.mean(distances),
        std_distance=np.std(distances),
        rewards=rewards,
        distances=distances,
        routes=routes,
    )
