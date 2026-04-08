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
    # x = np.arange(1, 101)
    # l1 = read('reward_forICM_DRQN_forReward14112024_3.txt', 100)
    # l2 = read('reward_forICM_DRQN_ICM_test_14112024_3.txt', 100)
    # plt.plot(x, l1, '-', color='b', label='DRQN')
    # plt.plot(x, l2, '-', color='r', label='DRQN with ICM')
    # plt.legend()
    # # plt.xticks(x[::1])
    # # plt.grid(True, linestyle='--', alpha=0.8)
    # font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    # plt.xlabel('Episodes', font)
    # plt.ylabel('Reward', font)
    # plt.show()

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=14)

# 动态障碍物数量测试
#     plt.figure(figsize=(7,5.6))
#     x = np.arange(3, 8, 1)
#     l2 = read('DRQN_successrate.txt', 5)
#     l3 = read('D3QN_24092024_T1_OB_successrate.txt', 5)
#     l1 = read('DRQN_ICM_successrate_2.txt', 5)
#     # l4 = read('D3QN_24092024_T1_OB_successrate.txt', 6)
#     plt.ylim(0.7, 1)
#     plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
#     plt.plot(x, l2, 'o-', color='b', label='DRQN')
#     # plt.plot(x, l4, 'o-', color='m', label='D3QN')
#     plt.plot(x, l3, 'o-', color='g', label='D3QN')
#
#     plt.legend()
#     plt.legend(loc='lower left')
#     plt.xticks(x[::1])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
#     plt.xlabel('Number of moving obstacles', font)
#     plt.ylabel('Success rate', font)
#     plt.show()
#
#     plt.figure(figsize=(7,5.6))
#     x = np.arange(3, 8, 1)
#     l2 = read('DRQN_avgtime.txt', 5)
#     l3 = read('D3QN_24092024_T1_OB_avgtime.txt', 5)
#     l1 = read('DRQN_ICM_avgtime_2.txt', 5)
#     # l4 = read('D3QN_16092024_R2_1_3OB_avgtime.txt', 6)
#     plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
#     plt.plot(x, l2, 'o-', color='b', label='DRQN')
#     # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
#     plt.plot(x, l3, 'o-', color='g', label='D3QN')
#
#     plt.ylim(200, 280)
#     plt.legend(loc='lower right')
#     plt.xticks(x[::1])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
#     plt.xlabel('Number of moving obstacles', font)
#     plt.ylabel('Average time (s)', font)
#     plt.show()
#
#     plt.figure(figsize=(7,5.6))
#     x = np.arange(3, 8, 1)
#     l2 = read('DRQN_TUserverate.txt', 5)
#     l3 = read('D3QN_24092024_T1_OB_TUserverate.txt', 5)
#     l1 = read('DRQN_ICM_TUserverate_2.txt', 5)
#     # l4 = read('D3QN_16092024_R2_1_3OB_TUserverate.txt', 6)
#     plt.ylim(0.95, 1)
#     plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
#     plt.plot(x, l2, 'o-', color='b', label='DRQN')
#     # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
#     plt.plot(x, l3, 'o-', color='g', label='D3QN')
#
#     plt.legend()
#     plt.xticks(x[::1])
#     plt.grid(True, linestyle='--', alpha=0.8)
#     font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
#     plt.xlabel('Number of moving obstacles', font)
#     plt.ylabel('Data collection rate', font)
#     plt.show()

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

# 节点数量测试
    plt.figure(figsize=(7,5.6))
    x = np.arange(6, 25, 3)
    l1 = read('DRQN_ICM_14112024_TU_T1_successrate.txt', 7)
    l2 = read('D3QN_forICM_17112024_TU_T1_successrate.txt', 7)
    l3 = read('DRQN_14112024_TU_T1_successrate.txt', 7)
    # l4 = read('D3QN_16092024_R2_2_3TU_successrate.txt', 21)
    plt.ylim(0.6, 1)
    plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    plt.plot(x, l3, 'o-', color='b', label='DRQN')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')

    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    plt.xlabel('Number of IoT sensor nodes', font)
    plt.ylabel('Success rate', font)
    plt.show()

    plt.figure(figsize=(7,5.6))
    x = np.arange(6, 25, 3)
    l1 = read('DRQN_ICM_14112024_TU_T1_avgtime.txt', 7)
    l2 = read('D3QN_forICM_17112024_TU_T1_avgtime.txt', 7)
    l3 = read('DRQN_14112024_TU_T1_avgtime.txt', 7)
    # l4 = read('D3QN_16092024_R2_2_3TU_avgtime.txt', 21)
    plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    plt.plot(x, l3, 'o-', color='b', label='DRQN')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')

    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    plt.xlabel('Number of IoT sensor nodes', font)
    plt.ylabel('Average time (s)', font)
    plt.show()

    plt.figure(figsize=(7,5.6))
    x = np.arange(6, 25, 3)
    l1 = read('DRQN_ICM_14112024_TU_T1_TUserverate.txt', 7)
    l2 = read('D3QN_forICM_17112024_TU_T1_TUserverate.txt', 7)
    l3 = read('DRQN_14112024_TU_T1_TUserverate.txt', 7)
    # l4 = read('D3QN_16092024_R2_2_3TU_TUserverate.txt', 21)
    plt.ylim(0.95, 1)
    plt.plot(x, l1, 'o-', color='r', label='ICM-DRQN')
    # plt.plot(x, l4, 'o-', color='m', label='DDQN-[22]')
    plt.plot(x, l3, 'o-', color='b', label='DRQN')
    plt.plot(x, l2, 'o-', color='g', label='D3QN')

    plt.legend()
    plt.xticks(x[::1])
    plt.grid(True, linestyle='--', alpha=0.8)
    font = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    plt.xlabel('Number of IoT sensor nodes', font)
    plt.ylabel('Data collection rate', font)
    plt.show()



