# # DQN Hyperparameters
# DQNGAMMA = 0.99  # 折扣因子 γ
# LEARNING_RATE = 0.001  # 学习率，可以调整
# MEMORY_SIZE = 100000  # 记忆缓冲区大小
# BATCH_SIZE = 20  # 最小一组数据个数
# EXPLORATION_MAX = 1.0  # 最大探索率
# EXPLORATION_MIN = 0.01  # 最小探索率
# EXPLORATION_DECAY = 0.998  # 探索率下降，可以调整
# DQNUNITS = 24 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看

# # PPO Hyperparameters
# GAMMA = 0.99 #伽马参数
# LAM = 0.95 #lamda参数
# CLIP_RATIO = 0.2 #上下裁剪区间，不用调整
# LR = 0.001 #learning rate 学习率，可以尝试做调整
# EPOCHS = 10 # 一组个数，不用调整
# ENTROPY = 0.01 # 学习熵，如果要进一步做可以调整
# UNITS = 64 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看

# # Training
# TOTAL_EPISODES = 100
# TARGET_UPDATE_FREQ = 10

# #Testing

# BEST_SCORE = 400


#12.14增加了 TARGET_UPDATE_FREQ和一些调参记录
# DQN Hyperparameters
DQNGAMMA = 0.99  # 折扣因子 γ
LEARNING_RATE = 0.000226  # 学习率，可以调整0.001/0.0001/0.0005/0.005/0.0003
MEMORY_SIZE = 100000  # 记忆缓冲区大小
BATCH_SIZE = 128  # 最小一组数据个数
EXPLORATION_MAX = 1.0  # 最大探索率
EXPLORATION_MIN = 0.01  # 最小探索率
EXPLORATION_DECAY = 0.99907  # 探索率下降，可以调整 0.998-0.9995-0.9993
DQNUNITS = 64 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看
TARGET_UPDATE_FREQ = 1000   # How often (in episodes) to update the target network.
                                # Note: The original framework updated per step. We can adapt this.
#新增的参数：控制每隔多少轮游戏（episodes），就将主网络（Policy Net）的权重复制给目标网络（Target Net）。这是 DQN 算法稳定性的关键一步。

# PPO Hyperparameters
GAMMA = 0.99 #伽马参数0.99-0.95
LAM = 0.95 #lamda参数
CLIP_RATIO = 0.2 #上下裁剪区间，不用调整0.2-0.3-0.1
LR = 0.001 #learning rate 学习率，可以尝试做调整0.001-0.0005-0.005
EPOCHS = 10 # 一组个数，不用调整
ENTROPY = 0.01 # 学习熵，如果要进一步做可以调整
UNITS = 64 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看64-32-128

# Training
TOTAL_EPISODES = 500
TARGET_UPDATE_FREQ = 10

#Testing
BEST_SCORE = 400
