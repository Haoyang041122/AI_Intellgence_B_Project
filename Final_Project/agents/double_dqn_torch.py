
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import config

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, units: int):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, act_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DoubleDQNAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # --- THIS IS THE CORRECTED LINE ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- END OF CORRECTION ---
        
        self.exploration_rate = config.EXPLORATION_MAX
        self.step_count = 0

        self.policy_net = QNetwork(obs_dim, act_dim, units=config.DQNUNITS).to(self.device)
        self.target_net = QNetwork(obs_dim, act_dim, units=config.DQNUNITS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        self.memory = deque(maxlen=config.MEMORY_SIZE)

    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        if evaluation_mode or random.random() > self.exploration_rate:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state_t)
            return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.act_dim - 1)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state.squeeze(0), action, reward, next_state.squeeze(0), done))
        
        if len(self.memory) >= config.BATCH_SIZE:
            self._learn()

        self.exploration_rate = max(config.EXPLORATION_MIN, self.exploration_rate * config.EXPLORATION_DECAY)

        self.step_count += 1
        if self.step_count % config.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _learn(self):
        batch = random.sample(self.memory, config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        target_q = rewards + (1 - dones) * config.DQNGAMMA * next_q_values
        current_q = self.policy_net(states).gather(1, actions)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # “安全锁”，防止梯度爆炸导致模型崩盘
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        # Added weights_only=True to address the FutureWarning
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())