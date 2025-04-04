import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import copy
import math 
from PIL import Image
from shapely.geometry import Point, Polygon
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import scipy.ndimage as ndimage

class Agent(object):
    def __init__(self):
        self.name = None
        self.position = None
        self.cash = False # 碰撞
        self.action = None
        self.action_space = None
        self.observation_space = None
        self.plane = None
        self.inobstacle = False
        self.inclip = False
           
class MultiEnvironment(gym.Env):
    def __init__(self):

        self.width = 100
        self.height = 60
        self.cash_a = False #画圈
        self.agent_nums = 3
        self.goal_num = 3
        self.agent_size = 2.5
        self.cash_distance = 5  #安全距离
        self.map_size = 0
        self.adversary = False   #有敌机
        self.design = False  # 画轨迹
        self.ep_length = 400
        # 实例化智能体
        self.agents = [Agent() for i in range(self.agent_nums)]
        self.rewards = []
        
        for index, agent in enumerate(self.agents):
            agent.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
            agent.observation_space = spaces.Box(low = -1, high= 1, shape=(13,),dtype=np.float32)
            agent.name = 'agent'
            agent.plane = np.array(Image.open("./common/plane.png"))
                              
        ######## 障碍物数据 ##############
        self.map =  pd.read_excel('common/obstacle.xlsx', sheet_name=None, header=None)['Sheet1']
     
        self.obstacle_pos = np.array([[self.map.values[:,0][1:6],self.map.values[:,1][1:6]],
                                      [self.map.values[:,3][1:6],self.map.values[:,4][1:6]],
                                      [self.map.values[:,6][1:6], self.map.values[:,7][1:6]],
                                      [self.map.values[:, 9][1:6], self.map.values[:, 10][1:6]]
                                      ], dtype=object)
        # 障碍物坐标对
        self.obstacle_cor = np.array([
            np.column_stack((self.obstacle_pos[0][0], self.obstacle_pos[0][1])),
            np.column_stack((self.obstacle_pos[1][0], self.obstacle_pos[1][1])),
            np.column_stack((self.obstacle_pos[2][0], self.obstacle_pos[2][1])),
            np.column_stack((self.obstacle_pos[3][0], self.obstacle_pos[3][1]))
                                      ], dtype=object)
        self.obstacle = Polygon(self.obstacle_cor[0])

        for i in range(1, len(self.obstacle_pos)):

            self.obstacle = self.obstacle.union(Polygon(self.obstacle_cor[i]))


        self.goal_1_position = np.array([90, 50], dtype=np.float32)
        self.goal_2_position = np.array([85, 45], dtype=np.float32)
        self.goal_3_position = np.array([98, 45], dtype=np.float32)

        # 固定位置
        self.corner_position = np.array([
            [-self.map_size, -self.map_size],
            [self.width + self.map_size, -self.map_size],
            [-self.map_size, self.height + self.map_size],
            [self.width + self.map_size, self.height + self.map_size,]
        ])

        self.color1 = [0.5, 0.2, 0.3]
        self.color2 = [0.25, 0.1, 0.25]
        # # 可视化显示

        self.fig, self.ax = plt.subplots()

    def reset(self):
        self.path = []
        self.paths = []
        self.times = 0
        # 初始化智能体位置
        for index, agent in enumerate(self.agents):
            agent.action = np.zeros([2])
            if index == 0:
                agent.position = np.array([15, 15], dtype=np.float32)  # 1初始位置
            if index == 1:
                agent.position = np.array([10, 10], dtype=np.float32)  # 2初始位置
            if index == 2:
                agent.position = np.array([10, 20], dtype=np.float32)  # 3初始位置
            self.path = [agent.position.copy()]
            self.paths.append(self.path)
        states = self._get_position()
        
        return states

    def step(self, actions):
        # dones = [False for i in range(self.agent_nums)]
        self.times += 1
        # 动作交互
        for index, agent in enumerate(self.agents):
            agent.position = agent.position + actions[index]
            self.paths[index].append(agent.position.copy())
            agent.action = actions[index]

        # 奖励函数
        rewards, dones, juli = self._get_reward()
        
        # # 状态裁剪
        for agent in self.agents:
            agent.position = np.clip(agent.position,[self.agent_size, self.agent_size], [self.width-self.agent_size, self.height-self.agent_size])# 不能超出边界
        # 更新状态
        states = self._get_position()
        self.render()
        return states, rewards, dones, juli, {}

    def render(self, mode='human'):

        self.ax.clear()
        self.ax.fill(self.obstacle_pos[0][0], self.obstacle_pos[0][1], color=self.color1)
        self.ax.fill(self.obstacle_pos[1][0], self.obstacle_pos[1][1], color=self.color1)
        self.ax.fill(self.obstacle_pos[2][0], self.obstacle_pos[2][1], color=self.color2)
        self.ax.fill(self.obstacle_pos[3][0], self.obstacle_pos[3][1], color=self.color2)

        # 智能体位置更新        
        for index, agent in enumerate(self.agents):
            
            angle = np.arctan2(agent.action[0], agent.action[1]) * 180 / np.pi
            rotated_plane_data = ndimage.rotate(agent.plane, angle, reshape=True)
            self.ax.imshow(rotated_plane_data, extent=[agent.position[0] - self.agent_size, agent.position[0] + self.agent_size, agent.position[1] - self.agent_size, agent.position[1]+self.agent_size])



        for corner in self.corner_position: # 画面固定
            self.ax.scatter(corner[0], corner[1], marker='o', color='white')
            
        # 轨迹更新
        if self.design:
            for index, path in enumerate(self.paths):
                if index == 0:
                    x, y = zip(*path)
                    self.ax.plot(x, y, 'r-', linewidth=1)
                if index == 1:
                    x, y = zip(*path)
                    self.ax.plot(x, y, 'b-', linewidth=1)
                if index == 2:
                    x, y = zip(*path)
                    self.ax.plot(x, y, 'g-', linewidth=1)

        self.fig.canvas.draw()
        plt.pause(0.01)

    def _get_position(self):
        # 智能体的位置 
        states = np.empty(self.agent_nums, dtype=object)
        positions = []
        for index, agent in enumerate(self.agents):
            positions.append(agent.position)
        for index, agent in enumerate(self.agents):
            other_position = np.delete(positions, index, axis=0) - positions[index]
            
            ############## 碰撞判断 ##################
            dis = np.min(np.sqrt(np.abs(other_position[:,-1]**2 + other_position[:,0]**2)))
            if dis < self.cash_distance:
                agent.cash = True
            else:
                agent.cash = False
            
            # 不能进入障碍物
            point = Point(agent.position)
            tag = point.within(self.obstacle)
            #global inobstacle
            if tag:
                agent.position -= agent.action
                agent.inobstacle = True
            else:
                agent.inobstacle = False
                
            # 边界裁剪#
            #print("0", agent.position)
            if agent.position[0] < self.agent_size or agent.position[0] > self.width-self.agent_size or agent.position[1] < self.agent_size or agent.position[1] > self.height-self.agent_size:
                agent.position = np.clip(agent.position,[self.agent_size, self.agent_size], [self.width-self.agent_size, self.height-self.agent_size])# 不能超出边界
                agent.inclip = True
            else:
                agent.inclip = False

            ############## 将各个状态信息放入state中 ##########
            state = np.array([])
            state = np.append(state, agent.position) # 智能体自身位置
            state = np.append(state, other_position.flatten())# 其他智能体相对位置
            if index == 0:
                goal_1_position = self.goal_1_position - positions[index]   # 目标与智能体的相对位置
                state = np.append(state, goal_1_position)
            if index == 1:
                goal_2_position = self.goal_2_position - positions[index]
                state = np.append(state, goal_2_position)
            if index == 2:
                goal_3_position = self.goal_3_position - positions[index]
                state = np.append(state, goal_3_position)

            state = np.append(state, agent.action)
            state = np.append(state, int(agent.inobstacle))  # 是否进入障碍物中
            state = np.append(state, int(agent.cash))   #我方智能体之间是否碰撞
            state = np.append(state, int(agent.inclip))  # 是否触碰边界
            states[index] = state

        return states
    
    def _caculate_distance(self):

        distances = np.empty((self.agent_nums), dtype=object)
        for index, agent in enumerate(self.agents):
            
            if index == 0:
                dis = np.sqrt(np.sum(np.square(agent.position - self.goal_1_position))) # 所有智能体和目标的距离
                distances[index] = dis
            if index == 1:
                dis = np.sqrt(np.sum(np.square(agent.position - self.goal_2_position))) # 所有智能体和目标的距离
                distances[index] = dis
            if index == 2:
                dis = np.sqrt(np.sum(np.square(agent.position - self.goal_3_position))) # 所有智能体和目标的距离
                distances[index] = dis

        return distances

    def veloc(self):
        velocs = np.empty((self.agent_nums), dtype=object)
        for index, agent in enumerate(self.agents):
            vel = np.sqrt(np.sum(np.square(agent.action)))  # 速度
            velocs[index] = vel

        return velocs
    def _get_reward(self):
        rewards = np.array([0.0, 0.0, 0.0])
        dones = np.array([False for i in range(self.agent_nums)])
        juli = np.array(self._caculate_distance())
        zhuangtai = np.array(self._get_position())
        for j in range(self.agent_nums):
            rewards[j] = juli[j] * -1 #距离目标距离越小奖励越大
            if juli[j] <= 1:
                rewards[j] += 100
            if zhuangtai[j][-3]:   #与障碍物之间碰撞，惩罚
                rewards[j] -= 10
            if zhuangtai[j][-2]:  # agent之间碰撞时，惩罚
                rewards[j] -= 2
            if zhuangtai[j][-1]:  # agent超出边界时，惩罚
                rewards[j] -= 2

        self.cash_a = False
        if np.max(np.abs(juli[:])) < 1:
            dones = np.array([False for i in range(self.agent_nums)])
        return rewards, dones, juli
        
if __name__ == "__main__":
    env = MultiEnvironment()
    states = env.reset()
    for episode in range(100):
        # actions = [env.action_space.sample() for i in range(4)]
        actions = [env.agents[i].action_space.sample() for i in range(4)]

        obs, reward, done, juli, info = env.step(actions)
        env.render()
        #print("step",i)
        if done[0]:
            env.reset()
    env.close()
