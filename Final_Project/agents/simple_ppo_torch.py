import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import config

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, units):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, units):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )

    def forward(self, x):
        return self.net(x)

class SimplePPO:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.best_score = 0

        # 从 config.py 读取参数
        self.gamma = config.GAMMA
        self.lam = config.LAM
        self.clip_ratio = config.CLIP_RATIO
        self.learning_rate = config.LR
        self.epochs = config.EPOCHS
        self.entropy_coef = config.ENTROPY
        self.units = config.UNITS

        # 构建网络
        self.actor = ActorNetwork(observation_space, action_space, self.units).to(device)
        self.critic = CriticNetwork(observation_space, self.units).to(device)
        
        # 用于计算 ratio 的旧策略网络
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.eval() # 旧网络不需要训练，设为评估模式

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # 损失函数 (Critic 用 MSE)
        self.mse_loss = nn.MSELoss()

    def act(self, state):
        """
        根据当前状态选择动作
        """
        # 转换状态为 Tensor
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probabilities = self.actor(state_t)
        
        probs_np = probabilities.cpu().numpy()[0]
        
        # 根据概率分布随机采样动作
        action = np.random.choice(self.action_space, p=probs_np)
        return action, probs_np

    def compute_returns_advantages(self, rewards, values, dones, next_value=0):
        """
        计算回报和优势函数 (GAE)
        """
        returns = []
        advantages = []
        gae = 0
        
        # 把 next_value 接在 values 后面方便计算
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            # Return = Advantage + Value
            returns.insert(0, gae + values[t])

        return np.array(returns), np.array(advantages)

    def train_episode(self, states, actions, rewards, dones):
        if len(states) < 4:
            return None, None

        # --- 1. 数据准备 ---
        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)

        # 转换为 Tensor
        states_t = torch.FloatTensor(states_np).to(device)
        actions_t = torch.LongTensor(actions_np).unsqueeze(1).to(device)
        
        # --- 2. 获取当前 Critic 预测值，用于计算 GAE ---
        with torch.no_grad():
            values_np = self.critic(states_t).cpu().numpy().flatten()
        
        # 计算 Returns 和 Advantages
        returns_np, advantages_np = self.compute_returns_advantages(
            rewards_np, values_np.tolist(), dones_np.tolist(), next_value=0
        )
        
        # 优势函数归一化 (稳定训练的关键)
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
        
        # 转回 Tensor 用于训练
        returns_t = torch.FloatTensor(returns_np).unsqueeze(1).to(device)
        advantages_t = torch.FloatTensor(advantages_np).unsqueeze(1).to(device)

        # --- 3. 更新旧策略网络 ---
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # 获取旧策略的概率分布 (Detach防止梯度反传)
        with torch.no_grad():
            old_probs = self.old_actor(states_t)
            # 获取实际执行动作的旧概率: gather 对应动作的概率
            old_action_probs = old_probs.gather(1, actions_t)

        actor_losses = []
        critic_losses = []

        # --- 4. PPO 核心训练循环 ---
        for _ in range(self.epochs):
            # === 更新 Actor ===
            # 计算新策略的概率
            new_probs = self.actor(states_t)
            new_action_probs = new_probs.gather(1, actions_t)
            
            # 计算比率 Ratio (New / Old)
            ratio = new_action_probs / (old_action_probs + 1e-8)
            
            # PPO Clip Loss
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy Bonus (鼓励探索)
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()
            
            # 总 Actor Loss
            actor_loss = policy_loss - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())

            # === 更新 Critic ===
            current_values = self.critic(states_t)
            critic_loss = self.mse_loss(current_values, returns_t)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)
    
    def save_model(self, filepath, score=None):
        if score is not None:
            base_path = f"{filepath}_score_{int(score)}"
        else:
            base_path = filepath
        os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else '.', exist_ok=True)
        
        # PyTorch 推荐保存 .pth 后缀
        path = f"{base_path}.pth"
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'old_actor': self.old_actor.state_dict()
        }, path)
        
        print(f"Saved model to: {path}")
    
    def load_model(self, filepath):
        # 兼容处理
        if not filepath.endswith('.pth'):
            path = f"{filepath}.pth"
        else:
            path = filepath
            
        checkpoint = torch.load(path, map_location=device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.old_actor.load_state_dict(checkpoint['old_actor'])
        
        print(f"Loaded model from: {path}")