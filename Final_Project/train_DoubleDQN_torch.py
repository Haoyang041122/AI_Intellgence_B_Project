"""
CartPole Training Script (Gymnasium Version)
--------------------------------------------
Main training loop that runs the CartPole-v1 environment using a DQN agent.
The training progress is logged and plotted using ScoreLogger.
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

# Import all agent classes
from agents.double_dqn_torch import DoubleDQNAgent
from scores.score_logger import ScoreLogger
import config

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

def train(algorithm: str = "doubledqn", terminal_penalty: bool = True):
    model_name = f"cartpole_{algorithm.lower()}.torch"
    model_path = os.path.join(MODEL_DIR, model_name)
    best_model_path = os.path.join(MODEL_DIR, f"best_{algorithm.lower()}.torch")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(f"{ENV_NAME}_{algorithm.upper()}_WithHighReward") # 在日志文件名里加个标记

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if algorithm.lower() == "doubledqn":
        agent = DoubleDQNAgent(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    print(f"[Info] Training with {algorithm.upper()} (High-Score Reward Enabled). Using device: {agent.device}")
    
    best_score = 0

    for run in range(1, config.TOTAL_EPISODES + 1):
        state, _ = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0
        done = False

        while not done:
            steps += 1
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            #基础惩罚
            if terminal_penalty and done and steps < 500:
                reward = -1.0
            
            # # 额外的高分奖励
            # if done and steps >= 500:
            #     reward += 2
                

            # # 在高分段每多活一步，也给点小甜头。
            # if steps > 400:
            #     reward += 0.1
           

            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

        epsilon = getattr(agent, 'exploration_rate', 0.0)
        print(f"Run: {run}, Epsilon: {epsilon:.3f}, Score: {steps}")
        logger.add_score(steps, run)

        if steps > best_score and steps >= config.BEST_SCORE:
            best_score = steps
            agent.save(best_model_path)
            print(f"*** New Best Score: {best_score}. Model saved to {best_model_path} ***")

    env.close()
    agent.save(model_path)
    print(f"[Train] Final model saved to {model_path}")
    return agent, model_path


def evaluate(model_path: str, algorithm: str, episodes: int = 100, render: bool = False, fps: int = 60):
    print(f"\n--- Starting Evaluation for {algorithm.upper()} ---")
    print(f"[Eval] Using model: {model_path}")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    if algorithm.lower() == "doubledqn":
        agent = DoubleDQNAgent(obs_dim, act_dim)
    else:
        raise ValueError(f"Evaluation not set up for {algorithm}")

    agent.load(model_path)
    print(f"[Eval] Loaded model weights.")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0
        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
            if dt > 0: time.sleep(dt)
        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores))
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


if __name__ == "__main__":
    algo_to_run = "doubledqn"
    _, final_model_path = train(algorithm=algo_to_run)
    
    evaluate(model_path=final_model_path, algorithm=algo_to_run)

    best_model_path = os.path.join(MODEL_DIR, f"best_{algo_to_run}.torch")
    if os.path.exists(best_model_path):
        evaluate(model_path=best_model_path, algorithm=algo_to_run)