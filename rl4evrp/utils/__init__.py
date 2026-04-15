"""
Utility functions for RL4EVRP: episode runner, training loops, etc.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from rl4evrp.environment import EVRPEnv


def run_episode(agent, inst: dict, device: str = 'cpu', greedy: bool = False,
                collect_traces: bool = False,
                bat_perturb: Optional[float] = None,
                cargo_perturb: Optional[float] = None) -> Tuple:
    """
    Run a single episode of EVRP with the given agent.
    
    Args:
        agent: A2CAgent instance
        inst: Instance dict from generate_instance()
        device: Device to use
        greedy: Whether to use greedy policy
        collect_traces: Whether to collect detailed per-step traces for XAI
        bat_perturb: Battery perturbation factor (for counterfactual analysis)
        cargo_perturb: Cargo perturbation factor (for counterfactual analysis)
    
    Returns:
        total_reward: Total episode reward
        route: List of visited nodes
        total_distance: Total distance traveled
        final_info: Dict with episode statistics
        transitions: List of transition dicts for training
        traces: (optional) List of per-step trace dicts
        env: The environment instance (for diagnostic logging)
    """
    env = EVRPEnv(inst, reward_mode=inst.get('reward_mode', 'distance'))
    obs = env.reset()
    done = False
    total_reward = 0.0
    transitions = []
    traces = [] if collect_traces else None
    use_grad = (not greedy) or collect_traces
    
    while not done:
        # Apply state perturbations (for counterfactual analysis)
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
        
        # Select action (with gradient tracking during training)
        with torch.set_grad_enabled(use_grad):
            action, log_prob, entropy, value = agent.select_action(obs_use, greedy=greedy)
        
        # Collect traces if requested
        if collect_traces:
            scores, _, _, dec_attn, enc_attn = agent._forward(obs_use, return_attn=True)
            probs = torch.softmax(scores.squeeze(0), -1)
            
            # Average attention over heads
            attn_avg = (
                dec_attn[0].mean(0).squeeze(0).detach().cpu().tolist()
                if dec_attn is not None else [0.0] * env.n
            )
            
            # Encoder self-attention at current node
            enc_attn_row = (
                enc_attn[0].mean(0)[obs['current_node']].detach().cpu().tolist()
                if enc_attn is not None else [0.0] * env.n
            )
            
            # Top-3 actions
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
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Store transition
        transitions.append(dict(
            reward=reward,
            log_prob=log_prob,
            entropy=entropy,
            value=value
        ))
        
        obs = next_obs
    
    return total_reward, env.route, env.total_d, info, transitions, traces, env


def train_agent(agent, train_instances: List[Dict], n_episodes: int = 100,
                device: str = 'cpu', save_dir: Optional[Path] = None,
                eval_instances: Optional[List[Dict]] = None,
                save_interval: int = 10) -> Dict:
    """
    Train A2C agent for specified number of episodes.
    
    Args:
        agent: A2CAgent instance
        train_instances: List of training instances
        n_episodes: Number of episodes to train
        device: Device to use
        save_dir: Directory to save checkpoints
        eval_instances: (optional) Instances for evaluation
        save_interval: Save checkpoint every N episodes
    
    Returns:
        results: Dict with training statistics
    """
    # Ensure agent is in training mode and set device
    agent.train()
    agent.to(device)
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    train_rewards = []
    eval_rewards = [] if eval_instances else None
    losses = []
    entropies = []
    
    for episode in range(n_episodes):
        # Select random instance
        inst = train_instances[episode % len(train_instances)]
        inst['reward_mode'] = 'distance'  # Default reward mode
        
        # Run episode with gradients enabled
        with torch.enable_grad():
            episode_reward, route, total_dist, info, transitions, _, _ = run_episode(
                agent, inst, device=device, greedy=False
            )
        
        # Update agent with gradients enabled
        with torch.enable_grad():
            loss, ent = agent.update([
                {
                    'r': t['reward'],
                    'lp': t['log_prob'],
                    'ent': t['entropy'],
                    'val': t['value']
                }
                for t in transitions
            ])
        
        train_rewards.append(episode_reward)
        losses.append(loss)
        entropies.append(ent)
        
        # Evaluation
        if eval_instances and (episode + 1) % save_interval == 0:
            eval_rews = []
            for eval_inst in eval_instances[:5]:  # Eval on first 5
                eval_inst['reward_mode'] = 'distance'
                eval_reward, _, _, _, _, _, _ = run_episode(
                    agent, eval_inst, device=device, greedy=True
                )
                eval_rews.append(eval_reward)
            
            avg_eval = np.mean(eval_rews)
            eval_rewards.append(avg_eval)
            
            print(f"Episode {episode+1}: train_reward={episode_reward:.3f}, "
                  f"eval_reward={avg_eval:.3f}, loss={loss:.4f}")
        else:
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}: train_reward={episode_reward:.3f}, loss={loss:.4f}")
        
        # Save checkpoint
        if save_dir and (episode + 1) % save_interval == 0:
            checkpoint_path = save_dir / f"agent_episode_{episode+1}.pt"
            torch.save(agent.state_dict(), checkpoint_path)
    
    return dict(
        train_rewards=train_rewards,
        eval_rewards=eval_rewards,
        losses=losses,
        entropies=entropies
    )


def evaluate_agent(agent, instances: List[Dict], device: str = 'cpu',
                   greedy: bool = True, n_eval: Optional[int] = None) -> Dict:
    """
    Evaluate agent on a set of instances.
    
    Args:
        agent: A2CAgent instance
        instances: Evaluation instances
        device: Device to use
        greedy: Whether to use greedy policy
        n_eval: Number of instances to evaluate (None = all)
    
    Returns:
        stats: Dictionary with evaluation statistics
    """
    n_eval = n_eval or len(instances)
    instances = instances[:n_eval]
    
    rewards = []
    distances = []
    routes = []
    
    for inst in instances:
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
        routes=routes
    )
