import matplotlib.pyplot as plt
import numpy as np

def save(pathname, list):
    fileOpen = open(pathname, 'w')

    for i in range(len(list)):
        fileOpen.write(str(list[i]))
        fileOpen.write('\n')
    fileOpen.close()

def read(pathname, length):
    fileOpen = open(pathname, 'r')
    list = np.zeros(length)
    for i in range(length):
        list[i] = fileOpen.readline()
    return list

if __name__ == '__main__':
    '''x = np.arange(0, 10050, 50)
    l1 = read('reward_test_5_5_3_ED3_t2_nTU_obv3.txt', 201)
    l2 = read('reward_test_5_5_3_D3_3cen_nTU_obv3.txt', 201)
    l3 = read('reward_test_5_5_3_DQN_3cen_nTU_obv3.txt',201)'''
# 动态障碍物数量测试
    plt.figure(1)
    x = np.arange(3, 8, 1)
    l1 = read('DRQN_successrate.txt', 5)
    l2 = read('D3QN_24092024_T1_OB_successrate.txt', 5)
    l3 = read('DRQN_ICM_successrate_2.txt', 5)
    # l4 = read('D3QN_24092024_T1_OB_successrate.txt', 6)
    plt.ylim(0.7, 1)
    plt.plot(x, l1, 'o-', color='b', label='DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='D3QN')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')
    plt.plot(x, l3, 'o-', color='r', label='DRQN with ICM')
    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    plt.xlabel('Number of moving obstacles', font)
    plt.ylabel('Success rate', font)
    plt.show()

    plt.figure(2)
    x = np.arange(3, 8, 1)
    l1 = read('DRQN_avgtime.txt', 5)
    l2 = read('D3QN_24092024_T1_OB_avgtime.txt', 5)
    l3 = read('DRQN_ICM_avgtime_2.txt', 5)
    # l4 = read('D3QN_16092024_R2_1_3OB_avgtime.txt', 6)
    plt.plot(x, l1, 'o-', color='b', label='DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')
    plt.plot(x, l3, 'o-', color='r', label='DRQN with ICM')
    # plt.ylim(0.9, 1)
    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    plt.xlabel('Number of moving obstacles', font)
    plt.ylabel('Average time (s)', font)
    plt.show()

    plt.figure(3)
    x = np.arange(3, 8, 1)
    l1 = read('DRQN_TUserverate.txt', 5)
    l2 = read('D3QN_24092024_T1_OB_TUserverate.txt', 5)
    l3 = read('DRQN_ICM_TUserverate_2.txt', 5)
    # l4 = read('D3QN_16092024_R2_1_3OB_TUserverate.txt', 6)
    plt.ylim(0.9, 1)
    plt.plot(x, l1, 'o-', color='b', label='DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')
    plt.plot(x, l3, 'o-', color='r', label='DRQN with ICM')
    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    plt.xlabel('Number of moving obstacles', font)
    plt.ylabel('Data collection rate', font)
    plt.show()

    # plt.figure(3)
    # x = np.arange(3, 9, 1)
    # l1 = read('DRQN_20092024_T1_collision_rate.txt', 6)
    # l2 = read('D3QN_24092024_data_OB_collision_rate.txt', 6)
    # l3 = read('D3QN_28092024_T4_OB_collision_rate.txt', 6)
    # # l4 = read('D3QN_16092024_R2_1_3OB_TUserverate.txt', 6)
    # plt.ylim(0, 0.1)
    # plt.plot(x, l1, 'o-', color='r', label='DRQN')
    # # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    # plt.plot(x, l2, 'o-', color='g', label='D3QN')
    # plt.plot(x, l3, 'o-', color='b', label='Dueling DQN')
    # plt.legend()
    # plt.xticks(x[::1])
    # plt.grid(True, linestyle='--', alpha=0.8)
    # font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    # plt.xlabel('Number of moving obstacles', font)
    # plt.ylabel('Collision rate', font)
    # plt.show()
#
# # 节点数量测试
#     plt.figure(1)
#     x = np.arange(5, 26, 1)
#     l1 = read('IoTs_testadd_reward0.005_0.01_successrate.txt', 21)
#     l2 = read('loTs_testD3QN_0.005_successrate.txt', 21)
#     l3 = read('loTs_testDQN_0.005_successrate.txt', 21)
#     l4 = read('D3QN_16092024_R2_2_3TU_successrate.txt', 21)
#     plt.ylim(0.6, 1)
#     plt.plot(x, l1, 'o-', color='r', label='RL-DC')
#     plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
#     plt.plot(x, l2, 'o-', color='g', label='D3QN-[31]')
#     plt.plot(x, l3, 'o-', color='b', label='DQN')
#     plt.legend()
#     plt.xticks(x[::5])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'simsun', 'weight': 'normal', 'size': 15}
#     plt.xlabel('物联网节点数量', font)
#     plt.ylabel('成功率', font)
#     plt.show()
#
#     plt.figure(2)
#     x = np.arange(5, 26, 1)
#     l1 = read('IoTs_testadd_reward0.005_0.01_avgtime.txt', 21)
#     l2 = read('loTs_testD3QN_0.005_avgtime.txt', 21)
#     l3 = read('loTs_testDQN_0.005_avgtime.txt', 21)
#     l4 = read('D3QN_16092024_R2_2_3TU_avgtime.txt', 21)
#     plt.plot(x, l1, 'o-', color='r', label='RL-DC')
#     plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
#     plt.plot(x, l2, 'o-', color='g', label='D3QN-[31]')
#     plt.plot(x, l3, 'o-', color='b', label='DQN')
#     plt.legend()
#     plt.xticks(x[::5])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'simsun', 'weight': 'normal', 'size': 15}
#     plt.xlabel('物联网节点数量', font)
#     plt.ylabel('平均任务时间/s', font)
#     plt.show()
#
#     plt.figure(3)
#     x = np.arange(5, 26, 1)
#     l1 = read('IoTs_testadd_reward0.005_0.01_TUserverate.txt', 21)
#     l2 = read('loTs_testD3QN_0.005_TUserverate.txt', 21)
#     l3 = read('loTs_testDQN_0.005_TUserverate.txt', 21)
#     l4 = read('D3QN_16092024_R2_2_3TU_TUserverate.txt', 21)
#     plt.ylim(0.92, 1)
#     plt.plot(x, l1, 'o-', color='r', label='RL-DC')
#     plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
#     plt.plot(x, l2, 'o-', color='g', label='D3QN-[31]')
#     plt.plot(x, l3, 'o-', color='b', label='DQN')
#     plt.legend()
#     plt.xticks(x[::5])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'simsun', 'weight': 'normal', 'size': 15}
#     plt.xlabel('物联网节点数量', font)
#     plt.ylabel('节点服务率', font)
#     plt.show()



