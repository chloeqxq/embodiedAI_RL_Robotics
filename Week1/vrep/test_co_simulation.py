import sys
import math
sys.path.append('../vrep')
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from class_OurQuadrupededRobot import class_quadrupededrobot
#学习
#  安装zmq文件 python -m pip install coppeliasim-zmqremoteapi-client
#  可能会更新一下pip  python.exe -m pip install --upgrade pip
#
#  pwd查看路径
#  安装后路径在D:\miniconda3\envs\PythonVrep\Lib\site-packages\coppeliasim_zmqremoteapi_client
# https://manual.coppeliarobotics.com/en/simulation.htm#stepping
# https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm
# https://manual.coppeliarobotics.com/en/apiFunctions.htm API句柄
# https://forum.coppeliarobotics.com/viewtopic.php?p=40377&hilit=sim.addForce#p40377 论坛

client = RemoteAPIClient()
print(client)
sim = client.require('sim')
sim.setStepping(True)
sim.startSimulation()

print('Program started')
#
robot_model = class_quadrupededrobot('QuadrupededRobot')
robot_model.initializehandle()#句柄初始化
while sim.getSimulationTime() < 5:
    robot_pose = robot_model.getrobot_pose()#获取机器人位置
    print('机器人实时位置',robot_pose[0:3])
    pi =math.pi
    qfl=[0, 0,  pi/3]
    qfr = [0, 0, 0]
    qbl = [0, 0, 0]
    qbr = [0, 0, 0]
    robot_model.set_joint_q(qfl, qfr, qbl, qbr)
    # bodyforce=[0,0,1]
    # robot_model.set_object_force(robot_pose[0:3],bodyforce)
    client.step() # 仿真运行一步
sim.stopSimulation()