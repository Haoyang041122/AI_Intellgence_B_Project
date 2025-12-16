from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
import time
import os
import config 

from agents.simple_ppo_torch import SimplePPO
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

# --- 内部故障模拟 ---
ACTION_NOISE_PROB = 0.0     #动作噪声-(0.1,0.15,0.2)
REWARD_NOISE_STD = 0.0      #奖励噪声-(1.0, 2.0)
SPARSE_REWARD = False       #稀疏奖励 (True)

# --- 外部攻击模拟 ---
# True开启
ADVERSARIAL_ATTACK = True
ATTACK_INTERVAL = 300       # 每隔多少步攻击一次(150-200-300)
ATTACK_FORCE = 2          # 攻击强度 (0.5-1-2)

def train_ppo_robust():
    suffix = ""
    if ACTION_NOISE_PROB > 0: suffix += f"_ActionNoise_{ACTION_NOISE_PROB}"
    if REWARD_NOISE_STD > 0: suffix += f"_RewardNoise_{REWARD_NOISE_STD}"
    if SPARSE_REWARD: suffix += "_Sparse"
    if ADVERSARIAL_ATTACK: suffix += f"_AdvAttack_F{ATTACK_FORCE}_I{ATTACK_INTERVAL}"
    
    if not suffix: suffix = "_Baseline"
        
    logger_name = f"{ENV_NAME}_PPO{suffix}"
    score_logger = ScoreLogger(logger_name)
    
    print(f"--- 启动 PPO 训练: {logger_name} ---")
    print(f"干扰设置: 动作噪声={ACTION_NOISE_PROB}, 奖励噪声={REWARD_NOISE_STD}, 稀疏奖励={SPARSE_REWARD}, 对抗攻击={ADVERSARIAL_ATTACK}")

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
            #动作干扰
            real_action = intended_action
            if np.random.rand() < ACTION_NOISE_PROB:
                real_action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(real_action)
            
            # 对抗性推力攻击
            if ADVERSARIAL_ATTACK and step > 0 and step % ATTACK_INTERVAL == 0:
                print(f"   ⚠️  Ep:{episode}, Step:{step} -> 遭受对抗性攻击！")
                current_env_state = list(env.unwrapped.state)
                
                # 随机选择攻击方向（拍杆子）
                direction = 1 if np.random.rand() > 0.5 else -1
                # 攻击角速度 (theta_dot)
                current_env_state[3] += direction * ATTACK_FORCE
                # 把被修改的状态写回环境
                env.unwrapped.state = tuple(current_env_state)
                # 更新 next_state
                next_state = np.array(current_env_state, dtype=np.float32)

            #奖励信号处理
            done = terminated or truncated

            if REWARD_NOISE_STD > 0:
                reward += np.random.normal(0, REWARD_NOISE_STD)
            
            if SPARSE_REWARD:#稀疏奖励
                if not done:
                    reward = 0
                else:
                    reward = 100 if step >= 490 else -1        

            episode_states.append(state)
            episode_actions.append(real_action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            state = next_state
            step += 1
            if done: break

        # if len(episode_states) > 0:
        #     agent.train_episode(episode_states, episode_actions, episode_rewards, episode_dones)
        if len(episode_states) > 0:
            for _ in range(3): #循环训练
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
