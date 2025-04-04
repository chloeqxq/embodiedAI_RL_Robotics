import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.require('sim')
# CartPole simulation model for VREP
class class_quadrupededrobot():
    def __init__(self, name='Innerclimbingscene_FourWheel'):
        """
        :param: name: string
            name of objective
        """
        super(self.__class__, self).__init__()
        self.name = name
        self.client_ID = None

        self.prismatic_joint_handle = None
        self.revolute_joint_handle = None

    def initializehandle(self):
        try:
            print('handles initialize')
        except:
            print('Failed handles initialize')

        # 句柄定义
        self.handle_world = sim.getObject('/Dummy_world') # 找到某一个关节
        self.handle_body= sim.getObject('/BODY_respondable')  # 找到某一个关节
        #
        #
        self.handle_FR_1A = sim.getObject('/FR_1A')
        self.handle_FR_2A  = sim.getObject('/FR_2A')
        self.handle_FR_3A  = sim.getObject('/FR_3A')
        self.handle_FR_4A  = sim.getObject('/FR_4A')
        self.handle_FL_1A = sim.getObject('/FL_1A')
        self.handle_FL_2A  = sim.getObject('/FL_2A')
        self.handle_FL_3A  = sim.getObject('/FL_3A')
        self.handle_FL_4A  = sim.getObject('/FL_4A')
        self.handle_BR_1A = sim.getObject('/BR_1A')
        self.handle_BR_2A  = sim.getObject('/BR_2A')
        self.handle_BR_3A  = sim.getObject('/BR_3A')
        self.handle_BR_4A  = sim.getObject('/BR_4A')
        self.handle_BL_1A = sim.getObject('/BL_1A')
        self.handle_BL_2A  = sim.getObject('/BL_2A')
        self.handle_BL_3A  = sim.getObject('/BL_3A')
        self.handle_BL_4A  = sim.getObject('/BL_4A')
        time.sleep(0.5)
    # =================== 获取状态 ==============================================
    def getrobot_pose(self):
        """
        :param: joint_name: string
        """
        body_position = sim.getObjectPosition(self.handle_body, self.handle_world)
        body_eulerAngles = sim.getObjectOrientation(self.handle_body, self.handle_world)
        body_pose = body_position+body_eulerAngles
        return body_pose

    def getrobot_velocity(self):
        linearVelocity, angularVelocity = sim.getObjectVelocity(self.handle_body)
        return linearVelocity, angularVelocity

    def get_FR_velocity(self):
        fr_1a = sim.getJointVelocity(self.handle_FR_1A)
        fr_2a = sim.getJointVelocity(self.handle_FR_2A)
        fr_3a = sim.getJointVelocity(self.handle_FR_3A)
        fr_4a = sim.getJointVelocity(self.handle_FR_4A)
        FR_velocity = [fr_1a,fr_2a,fr_3a,fr_4a]
        return FR_velocity

    def get_FL_velocity(self):
        fl_1a = sim.getJointVelocity(self.handle_FL_1A)
        fl_2a = sim.getJointVelocity(self.handle_FL_2A)
        fl_3a = sim.getJointVelocity(self.handle_FL_3A)
        fl_4a = sim.getJointVelocity(self.handle_FL_4A)
        FL_velocity = [fl_1a,fl_2a,fl_3a,fl_4a]
        return FL_velocity

    def get_BR_velocity(self):
        br_1a = sim.getJointVelocity(self.handle_BR_1A)
        br_2a = sim.getJointVelocity(self.handle_BR_2A)
        br_3a = sim.getJointVelocity(self.handle_BR_3A)
        br_4a = sim.getJointVelocity(self.handle_BR_4A)
        FR_velocity = [br_1a, br_2a, br_3a, br_4a]
        return FR_velocity

    def get_BL_velocity(self):
        bl_1a = sim.getJointVelocity(self.handle_BL_1A)
        bl_2a = sim.getJointVelocity(self.handle_BL_2A)
        bl_3a = sim.getJointVelocity(self.handle_BL_3A)
        bl_4a = sim.getJointVelocity(self.handle_BL_4A)
        BL_velocity = [bl_1a, bl_2a, bl_3a, bl_4a]
        return BL_velocity

    # def get_back_proximity_sensor(self):
    #     res, dist, point, obj, n = sim.readProximitySensor(self.handle_back_proximity_sensor)
    #     # return dist, point, obj

    def set_joint_dq(self, dqfl, dqfr, dqbl, dqbr):
        # vrep_sim.simxPauseCommunication(self.client_ID,1)
        sim.setJointTargetVelocity(self.handle_FL_1A, dqfl[0])
        sim.setJointTargetVelocity(self.handle_FL_2A, dqfl[1])
        sim.setJointTargetVelocity(self.handle_FL_3A, dqfl[2])
        # r
        sim.setJointTargetVelocity(self.handle_FR_1A, dqfr[0])
        sim.setJointTargetVelocity(self.handle_FR_2A, dqfr[1])
        sim.setJointTargetVelocity(self.handle_FR_3A, dqfr[2])
        #
        sim.setJointTargetVelocity(self.handle_BL_1A, dqbl[0])
        sim.setJointTargetVelocity(self.handle_BL_2A, dqbl[1])
        sim.setJointTargetVelocity(self.handle_BL_3A, dqbl[2])
        #
        sim.setJointTargetVelocity(self.handle_BR_1A, dqbr[0])
        sim.setJointTargetVelocity(self.handle_BR_2A, dqbr[1])
        sim.setJointTargetVelocity(self.handle_BR_3A, dqbr[2])

    def set_joint_q(self, qfl, qfr, qbl, qbr):
        # vrep_sim.simxPauseCommunication(self.client_ID,1)
        sim.setJointTargetPosition(self.handle_FL_1A, qfl[0])
        sim.setJointTargetPosition(self.handle_FL_2A, qfl[1])
        sim.setJointTargetPosition(self.handle_FL_3A, qfl[2])
        # r
        sim.setJointTargetPosition(self.handle_FR_1A, qfr[0])
        sim.setJointTargetPosition(self.handle_FR_2A, qfr[1])
        sim.setJointTargetPosition(self.handle_FR_3A, qfr[2])
        #
        sim.setJointTargetPosition(self.handle_BL_1A, qbl[0])
        sim.setJointTargetPosition(self.handle_BL_2A, qbl[1])
        sim.setJointTargetPosition(self.handle_BL_3A, qbl[2])
        #
        sim.setJointTargetPosition(self.handle_BR_1A, qbr[0])
        sim.setJointTargetPosition(self.handle_BR_2A, qbr[1])
        sim.setJointTargetPosition(self.handle_BR_3A, qbr[2])

    # def set_object_force(self, position, force):
    #     sim.addForce(self.handle_body, position, force)

