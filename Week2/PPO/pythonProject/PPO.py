import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from actioncriticNN import ActorCritic

# 定义PPO代理
#抓住三个重点找，reward At clip
class PPO:
    '''PPO算法'''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                 lmbda, epsilon, epochs, device):
        self.action_dim = action_dim
        self.actor_critic = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor_parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic_parameters(), lr=critic_lr)
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE参数
        self.epsilon = epsilon  # PPO截断范围
        self.epochs = epochs  # PPO的更新次数
        self.device = device

    def take_action(self, state):
        '''根据策略网络选择动作'''
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        dist = torch.distributions.Categorical(action_probs) #作用是创建以参数probs为标准的类别分布
        action = dist.sample()
        return action.item()

    def update(self, transition_dict):
        '''更新策略网络和价值网络'''

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)#改变奖励
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)  # 改变奖励
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算TD误差和优势函数
        # Generalized Advantage Estimation (GAE) 计算广义优势估计（GAE）影响At

        _, state_values = self.actor_critic(states)
        _, next_state_values = self.actor_critic(next_states)
        td_target = rewards + self.gamma * next_state_values * (1 - dones) # 优势函数，计算当前动作好坏
        delta = td_target - state_values
        delta = delta.detach().cpu().numpy()

        advantage_list = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_list.append([advantage])
        advantage_list.reverse()
        advantages = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 计算旧策略的log概率
        with torch.no_grad():
            action_probs_old, _ = self.actor_critic(states)
            dist_old = torch.distributions.Categorical(action_probs_old)
            log_probs_old = dist_old.log_prob(actions)

        # 更新策略网络和价值网络
        for _ in range(self.epochs):
            action_probs, state_values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages.squeeze()
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.squeeze()
            # 计算TD误差和优势函数
            # 截断处理
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = F.mse_loss(state_values, td_target.detach())

            # 更新策略网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新价值网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
