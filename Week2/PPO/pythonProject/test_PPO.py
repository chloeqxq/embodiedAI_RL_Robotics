import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.hipify.hipify_python import hip_header_magic
from tqdm import tqdm  # 用于进度条显示
from PPO import PPO

# 训练PPO代理
def train_ppo():
    # 超参数设置
    actor_lr = 1e-3
    critic_lr = 1e-3
    num_episodes = 300
    hidden_dim = 128
    gamma = 0.99
    lmbda = 0.95
    epsilon = 0.3
    epochs = 20
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 环境初始化
    env_name = 'CartPole-v1'
    # env = gym.make(env_name, render_mode='human')  # 训练时不渲染，加速训练
    env = gym.make(env_name, render_mode='None')  # 训练时不渲染，加速训练
    env.reset(seed=0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                lmbda, epsilon, epochs, device)

    return_list = []  # 存储每个episode的回报

    for i in range(10):#回合之上总的回合个数
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i + 1}') as pbar:
            for i_episode in range(int(num_episodes / 10)):#回合
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, info = env.reset(seed=i * int(num_episodes / 10) + i_episode)
                done = False
                truncated = False

                while not (done or truncated):#每步奖励
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_return += reward

                    # 记录数据
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state

                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:])
                    pbar.set_postfix({
                        'Episode': f'{i * int(num_episodes / 10) + i_episode + 1}/{num_episodes}',
                        'Average Return': f'{avg_return:.2f}',
                    })
                pbar.update(1)

    env.close()

    # 绘制学习曲线和滑动平均曲线
    import matplotlib.pyplot as plt

    # 计算滑动平均
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    window_size = 10  # 滑动窗口大小
    moving_avg = moving_average(return_list, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(return_list, label='Return per Episode')
    # plt.plot(range(window_size - 1, num_episodes), moving_avg, label=f'Moving Average (window={window_size})',
    #          color='red')
    plt.plot(range(window_size - 1, num_episodes), moving_avg, label=f'Moving Average',
             color='red')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('PPO on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_ppo()
