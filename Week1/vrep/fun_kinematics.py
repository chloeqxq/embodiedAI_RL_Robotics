r_wheel = 0.1175
w_mp = 0.6

def mobilerobot_FK(mr_dq_br,mr_dq_bl):
    mr_v_bl= mr_dq_bl * r_wheel
    mr_v_br = mr_dq_br * r_wheel

    mr_v = (mr_v_bl + mr_v_br)/2
    mr_d_gamma = (mr_v_bl - mr_v_br)/w_mp
    mr_R = (w_mp * (mr_v_bl + mr_v_br)) / (2 * (mr_v_bl - mr_v_br))
    return mr_v, mr_d_gamma, mr_R

def mobilerobot_IK(mr_v,mr_d_gamma):
    mr_v_br = (w_mp * mr_d_gamma + 2 *mr_v)/ 2
    mr_v_bl = 2 * mr_v - mr_v_br

    mr_dq_br = mr_v_br/r_wheel
    mr_dq_bl = mr_v_bl/r_wheel

    return mr_dq_br, mr_dq_bl

def test_FK():
    print('运动学正解速度、角速度和转弯半径为',mobilerobot_FK(1, 2))

if __name__ == '__main__':
    test_FK()# 只是测试用，确保另外py文件调用fun_kinematics时不调用test_FK函数
