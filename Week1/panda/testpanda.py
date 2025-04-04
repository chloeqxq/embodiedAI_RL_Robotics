import gymnasium as gym
import panda_gym

env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
import gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer

env = gym.make("PandaReach-v2", render=True)
model = DDPG.load('ddpg_panda_reach_v2', env=env)
# model = TD3.load('td3_panda_reach_v2', env=env)
# model = SAC.load('sac_panda_reach_v2', env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print('Done')
        obs = env.reset()