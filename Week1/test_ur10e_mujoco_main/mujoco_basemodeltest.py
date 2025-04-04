import matplotlib.pyplot as plt
import mujoco
import glfw
import mujoco.viewer as viewer
import mujoco_viewer # pip install mujoco_python_viewer
from sympy.physics.units import angular_mil

# model = mujoco.MjModel.from_xml_path('./robot/ur10e.xml')
model = mujoco.MjModel.from_xml_path('./robot/scene.xml')
model_path = './robot/scene.xml'
data = mujoco.MjData(model)
print('data',data)
print('datatype',type(data))

# 获取所有关节名称
joint_names = []
for i in range(model.njnt):
    joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
# 显示所有关节名称
print("所有关节名称如下：")
for name in joint_names:
    print(name)
body_names = []
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_names.append(body_name)
# 显示所有 body 名称
print("所有 body 名称如下：")
for name in body_names:
    print(name)

# viewer.launch_from_path(model_path) #打开环境查看


#将状态创建为关键帧0
mujoco.mj_resetDataKeyframe(model,data,0)
#创建并运行查看器
mjviewer = mujoco_viewer.MujocoViewer(model,data)
# 模拟时间
duration = 10
# 定义数据采集列表
timevals = []
angular_vel = []
robot_hig = []
# 设置关节属性
mujoco.mj_resetData(model,data)
data.joint('shoulder_lift_joint').qvel = 20
# 开始模拟收集数据
while data.time < duration:
    mujoco.mj_step(model,data,nstep=5)
    mjviewer.render()

    timevals.append(data.time)
    angular_vel.append(data.qvel[0:3].copy())
    robot_hig.append(data.qpos[0:3].copy())

mjviewer.close()
# 关闭查看器
# 绘制图形
dpi = 120
width = 600
height = 800
figsize = (width/dpi, height/dpi)
_, ax = plt.subplots(2,1,figsize = figsize, dpi=dpi)

ax[0].plot(timevals,angular_vel)
ax[0].set_ylabel('qvel')
ax[1].plot(timevals,robot_hig)
ax[0].set_ylabel('qpos')

plt.show()

render = mujoco.Renderer(model)
render.update_scene(data)
image = render.render()
mujoco.mj_kinematics(model.data)