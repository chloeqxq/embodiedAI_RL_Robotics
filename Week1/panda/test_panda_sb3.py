import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO,A2C

env = gym.make('PandaReach-v3', render_mode="human")
model = PPO("MultiInputPolicy",
            env,
            verbose=1,
            n_steps=1024,
            )
model1 = A2C("MultiInputPolicy",
            env,
            verbose=1,
            n_steps=1024,
            )
model1.learn(total_timesteps=100)

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print('Done')
#         obs = env.reset()