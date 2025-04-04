import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm  # 用于进度条显示
# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    '''包含策略网络和价值网络的Actor-Critic模型'''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 策略网络
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)
        # 价值网络
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 策略网络前向传播
        actor_x = F.relu(self.actor_fc1(x)) #relu激活函数对fc1进行激活
        action_probs = F.softmax(self.actor_fc2(actor_x), dim=1)
        # 价值网络前向传播
        critic_x = F.relu(self.critic_fc1(x))
        state_values = self.critic_fc2(critic_x)
        return action_probs, state_values

    def actor_parameters(self):
        '''返回策略网络的参数'''
        return list(self.actor_fc1.parameters()) + list(self.actor_fc2.parameters())

    def critic_parameters(self):
        '''返回价值网络的参数'''
        return list(self.critic_fc1.parameters()) + list(self.critic_fc2.parameters())