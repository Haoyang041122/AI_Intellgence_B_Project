"""
CartPole PPO Training & Evaluation (PyTorch) - Final Logic Fixed
----------------------------------------------------------------
1. Trains the PPO agent.
2. Automatically evaluates BOTH the 'Final Model' and 'Best Model' after training.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
import time
import os
import config

# Import the PyTorch PPO Agent
from agents.simple_ppo_torch import SimplePPO
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

def train_ppo():
    # Set up
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    os.makedirs(MODEL_DIR, exist_ok=True)
    score_logger = ScoreLogger(f"{ENV_NAME}_PPO")
    
    agent = SimplePPO(observation_space, action_space)
    print(f"[Info] Starting PPO training (PyTorch)... Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    start_time = time.time()
    best_score = config.BEST_SCORE
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 路径定义
    final_model_path = os.path.join(MODEL_DIR, "cartpole_ppo_final.pth")
    # 定义最佳模型的保存路径
    best_model_base_path = os.path.join(MODEL_DIR, "best_ppo")
    best_model_full_path = None 

    scores_window = [] # 用于计算最近100局平均分

    # Training loop
    for episode in range(1, config.TOTAL_EPISODES + 1):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        step = 0
        while step < 500:
            action, prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            state = next_state
            step += 1

            if done:
                break

        # On-Policy Training
        if len(episode_states) > 0:
            actor_loss, critic_loss = agent.train_episode(
                episode_states, episode_actions, episode_rewards, episode_dones
            )
        else:
            actor_loss, critic_loss = 0, 0

        score = len(episode_rewards)
        scores_window.append(score)
        if len(scores_window) > 100: scores_window.pop(0)
        avg_score = np.mean(scores_window)
        
        #保存最佳模型
        if score >= best_score:
            best_score = score
            # 保存时会自动加上 _score_XXX.pth
            agent.save_model(best_model_base_path, score=score)
            # 记录下这个文件名，以便后面评估用
            best_model_full_path = f"{best_model_base_path}_score_{int(score)}.pth"
            print(f"*** New Best Score: {best_score} ***")

        # --- 日志 ---
        if episode % 10 == 0:
            print(f"Ep: {episode:4d}, Score: {score:3d}, Avg100: {avg_score:.2f}")
        
        # 记录到 CSV/PNG
        score_logger.add_score(score, episode)


    env.close()

    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time / 60:.1f} minutes")
    
    # 保存最终模型
    agent.save_model(os.path.join(MODEL_DIR, "cartpole_ppo_final")) # 这里的参数不带后缀
    
    # 返回: Agent对象, 最终模型路径, 最佳模型路径
    return agent, final_model_path, best_model_full_path


def evaluate(model_path: str, episodes: int = 100, render: bool = False, fps: int = 60):
    """
    Standard Evaluation Function with Deterministic Policy
    """
    if model_path is None or not os.path.exists(model_path):
        print(f"[Eval Error] Model path not found: {model_path}")
        return

    print(f"\n--- Starting Evaluation: {os.path.basename(model_path)} ---")
    
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = SimplePPO(obs_dim, act_dim)
    agent.load_model(model_path)
    
    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        done = False
        steps = 0
        
        while not done:
            # 关键：评估时使用确定性策略 (Argmax)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.actor.net[0].weight.device)
            with torch.no_grad():
                probs = agent.actor(state_t).cpu().numpy()[0]
            action = np.argmax(probs)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            steps += 1

            if dt > 0: time.sleep(dt)

        scores.append(steps)
        if ep % 10 == 0:
            print(f"[Eval] Episode {ep}/{episodes}: steps={steps}")

    env.close()
    avg = float(np.mean(scores))
    print(f"[Result] Average Score over {episodes} episodes: {avg:.2f}")
    print("-" * 50)
    return scores


if __name__ == "__main__":
    # 1. 运行训练，并获取路径
    # 注意：这里会返回两个路径：最终的和最佳的
    trained_agent, final_path, best_path = train_ppo()
    
    print("\n\n" + "="*30)
    print("TRAINING FINISHED. STARTING EVALUATION.")
    print("="*30)

    # 评估最终模型 (Final Model)
    evaluate(model_path=final_path, episodes=100, render=False)

    # 评估最佳模型 (Best Model)
    if best_path:
        evaluate(model_path=best_path, episodes=100, render=False)
    else:
        print("[Info] No Best Model saved (maybe score never hit threshold).")