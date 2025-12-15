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

# 鲁棒性设置 

# 场景 1: 动作干扰 -测试0.1/0.15/0.2
ACTION_NOISE_PROB = 0.0

# 场景 2: 奖励噪声 (给奖励加上高斯噪声)
# 0.0 = 无干扰, 测试1.0/2.0
REWARD_NOISE_STD = 0.0

# 场景 3: 稀疏奖励 (只有最后一步给分，中间全0)
# False = 正常
SPARSE_REWARD = True 

def train_ppo_robust():
    suffix = ""
    if ACTION_NOISE_PROB > 0: suffix += f"_ActionNoise_{ACTION_NOISE_PROB}"
    if REWARD_NOISE_STD > 0: suffix += f"_RewardNoise_{REWARD_NOISE_STD}"
    if SPARSE_REWARD: suffix += "_Sparse"
    
    logger_name = f"{ENV_NAME}_PPO{suffix}"
    score_logger = ScoreLogger(logger_name)
    
    print(f"--- 启动鲁棒性训练: {logger_name} ---")
    print(f"干扰设置: 动作噪声={ACTION_NOISE_PROB}, 奖励噪声={REWARD_NOISE_STD}, 稀疏奖励={SPARSE_REWARD}")

    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = SimplePPO(observation_space, action_space)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = 0

    for episode in range(1, config.TOTAL_EPISODES + 1):
        state, _ = env.reset()
        episode_states, episode_actions, episode_rewards, episode_dones = [], [], [], []
        step = 0
        
        while step < 500:
            intended_action, prob = agent.act(state)
            
            #动作干扰 (Action Attack)
            real_action = intended_action
            if np.random.rand() < ACTION_NOISE_PROB:
                # 强制随机动作 
                real_action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated

            # 奖励干扰
            if REWARD_NOISE_STD > 0:
                # 加上正态分布噪声 
                noise = np.random.normal(0, REWARD_NOISE_STD)
                reward += noise
            

            # 稀疏奖励 (Sparse Reward)
            if SPARSE_REWARD:
                if not done:
                    # 只要游戏没结束，一律给 0 分
                    reward = 0
                else:
                    if step >= 490: 
                        reward = 100 
                    else:
                        reward = -1        

            episode_states.append(state)
            episode_actions.append(real_action) # 记录实际发生的动作
            episode_rewards.append(reward)      # 记录被污染的奖励
            episode_dones.append(done)

            state = next_state
            step += 1
            if done: break

        if len(episode_states) > 0:
            agent.train_episode(episode_states, episode_actions, episode_rewards, episode_dones)

        score = step 
        score_logger.add_score(score, episode)
        
        if score >= best_score:
            best_score = score
            agent.save_model(f"{checkpoint_dir}/best_robust_model")

        if episode % 10 == 0:
            print(f"Ep: {episode}, Score: {score}, Best: {best_score}")

    env.close()
    return agent

if __name__ == "__main__":
    train_ppo_robust()