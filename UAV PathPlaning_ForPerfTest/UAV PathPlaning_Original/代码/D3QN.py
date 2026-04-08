import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *
from tqdm import tqdm
from save_data import *
import math

env = ENV()
env.reset()

NUM_STATES = 2 + 3*env.num_tu_can + 2*2 + env.num_lidar
NUM_ACTIONS = env.num_actions
LEARNING_RATE = 0.0001     #学习率
GAMMA = 0.95 #折扣因子
MEMORY_CAPACITY = 80000     #经验池容量
TRAIN_THRESHOLD = 30000     #训练开始阈值
BATCH_SIZE = 128    #单次学习取记录数
EPS_START = 0.8
EPS_END = 0.1
EPS_DECAY = 5000
Q_NETWORK_ITERATION = 100     #学习多少次后统一两个网络的参数

class QNetWork(nn.Module):
    def __init__(self):
        super(QNetWork, self).__init__()

        self.fc1 = nn.Linear(NUM_STATES, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2_v = nn.Linear(256, 128)
        self.fc2_v.weight.data.normal_(0, 0.1)
        self.fc2_a = nn.Linear(256, 128)
        self.fc2_a.weight.data.normal_(0, 0.1)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc3.weight.data.normal_(0, 0.1)
        #self.fc4 = nn.Linear(256, 256)
        #self.fc4.weight.data.normal_(0, 0.1)

        self.V = nn.Linear(128, 1)
        self.V.weight.data.normal_(0, 0.1)
        self.A = nn.Linear(128, NUM_ACTIONS)
        self.A.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x_v = torch.relu(self.fc2_v(x))
        x_a = torch.relu(self.fc2_a(x))
        # x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))

        V = self.V(x_v)
        A = self.A(x_a)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q

class D3QN():
    def __init__(self):
        self.local_net = QNetWork()
        self.target_net = QNetWork()

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=LEARNING_RATE)

    def choose_action_train(self, state, step_done):
        #print('state : {}'.format(state))
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        # print('torch state : {}'.format(state))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_done / EPS_DECAY)
        #print(eps_threshold)
        if np.random.randn() < eps_threshold:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action_value = self.local_net.forward(state).detach()
            action = int(torch.max(action_value, 1)[1].data.numpy())
        return action

    def choose_action_apply(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        #将扩展后的状态输入到本地网络（local_net）中进行前向传播，得到每个动作对应的 Q 值。
        action_value = self.local_net.forward(state).detach()
        action = int(torch.max(action_value, 1)[1].data.numpy())
        return action
    #将一个时间步的状态转换存储到经验回放缓存中。
    def store_transition(self, state, action, reward, next_state):
        index = self.memory_counter % MEMORY_CAPACITY
        for i in range(NUM_STATES):
            self.memory[index, i] = state[i]
        self.memory[index, NUM_STATES] = action
        self.memory[index, NUM_STATES + 1] = reward
        for i in range(NUM_STATES):
            self.memory[index, NUM_STATES + 2 + i] = next_state[i]
        self.memory_counter += 1

    def learn(self):
        #统一网络参数
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.local_net.state_dict())
        self.learn_step_counter += 1

        #在[0,MEMORY_CAPACITY)内输出BATCH_SIZE个数字
        sample_index = np.random.choice(
            self.memory_counter if self.memory_counter < MEMORY_CAPACITY else MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])
        q_local = self.local_net(batch_state).gather(1, batch_action)

        #DQN
        #q_next = self.target_net(batch_next_state).detach()

        #DDQN
        local_action_value = self.local_net.forward(batch_next_state).detach()
        local_action = torch.max(local_action_value, 1)[1].data.numpy()
        local_action_t = []
        for i in range(BATCH_SIZE):
            local_action_t.append([local_action[i]])
        local_action_t = np.array(local_action_t, ndmin=2)[:, :]
        local_action = torch.LongTensor(local_action_t.astype(int))
        q_next = self.target_net(batch_next_state).gather(1, local_action).detach()

        #这里有问题，如果next state为中止状态，q值应为batch_reward?
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_local, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_net(self, pathname):
        torch.save(self.local_net.state_dict(), 'localnet' + pathname)
        #torch.save(self.target_net.state_dict(), 'targetnet' + pathname)

    def load_net(self, pathname_loaclnet):
        self.local_net.load_state_dict(torch.load(pathname_loaclnet))
        self.target_net.load_state_dict(self.local_net.state_dict())


def state_debug(state):
    state_t = []
    if len(np.array(state, ndmin=2)[0]) > 1:
        for l in range(NUM_STATES):
            state_t.append(state[l])
    else:
        for l in range(NUM_STATES):
            state_t.append(state[l][0])

    return state_t

def train(file_name):
    d3qn = D3QN()
    episodes = 900
    print("Collecting Experience ....")
    #env.seed(250)
    count = 0
    step = 0
    successrate = []
    allsuccessrate = []
    stepdone_avg = []
    TUserverate = []
    reward_list = []
    reward_test_avg = []
    for _ in enumerate(tqdm(range(episodes))):
        state = env.reset()

        ep_reward = 0
        while 1:
            action = []
            for i in range(env.n_uav):
                if env.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(d3qn.choose_action_train(state[i], step))
                #如果达到了训练阈值（TRAIN_THRESHOLD）并且步数（step）被10整除，则调用D3QN模型的学习方法（d3qn.learn()）进行模型参数的更新。
                if d3qn.memory_counter >= TRAIN_THRESHOLD and step%10 == 0:
                    d3qn.learn()
                step += 1
            next_state, reward, done = env.step(action)
            #存储每个时间步的状态转换（state transition）到经验回放缓存中
            for i in range(env.n_uav):
                if action[i] != -1:
                    d3qn.store_transition(state[i], action[i], reward[i], next_state[i])
                    ep_reward += reward[i]

            if done:
                break
            state = next_state

        reward_list.append(ep_reward)

        if d3qn.memory_counter < TRAIN_THRESHOLD:
            continue

        '''num_try = 20
        if count % 50 == 0:
            r, sr, asr, tsr, s, t = performance_test(d3qn, num_try)
            successrate.append(sr)
            allsuccessrate.append(asr)
            TUserverate.append(tsr)
            stepdone_avg.append(s)
            reward_test_avg.append(r)
            save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)
            netname = file_name + '.pth'
            d3qn.store_net(netname)
        count += 1'''

    print('finished')
    r, sr, asr, tsr, s, t = performance_test(d3qn, 100)
    successrate.append(sr)
    allsuccessrate.append(asr)
    TUserverate.append(tsr)
    stepdone_avg.append(s)
    reward_test_avg.append(r)
    netname = file_name + '.pth'
    d3qn.store_net(netname)
    save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)

def train_multiuavs(file_name):
    d3qn = []
    for _ in range(env.n_uav):
        d3qn.append(D3QN())
    episodes = 2000
    print("Collecting Experience ....")
    #env.seed(250)
    count = 0
    step = 0
    successrate = []
    allsuccessrate = []
    stepdone_avg = []
    TUserverate = []
    reward_list = []
    reward_test_avg = []
    for _ in enumerate(tqdm(range(episodes))):
        state = env.reset()

        ep_reward = np.zeros(shape=(env.n_uav))
        while 1:
            action = []
            for i in range(env.n_uav):
                if env.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(d3qn[i].choose_action_train(state[i], step))

                if d3qn[i].memory_counter >= TRAIN_THRESHOLD and step%10 == 0:
                    d3qn[i].learn()
                step += 1
            next_state, reward, done = env.step(action)
            for i in range(env.n_uav):
                if action[i] != -1:
                    d3qn[i].store_transition(state[i], action[i], reward[i], next_state[i])
                    ep_reward[i] += reward[i]

            if done:
                break
            state = next_state

        reward_list.append(ep_reward) #三个智能体各自的reward

        if d3qn[0].memory_counter < TRAIN_THRESHOLD:
            continue

        '''num_try = 20
        if count % 50 == 0:
            r, sr, asr, tsr, s, t = performance_test(d3qn, num_try)
            successrate.append(sr)
            allsuccessrate.append(asr)
            TUserverate.append(tsr)
            stepdone_avg.append(s)
            reward_test_avg.append(r)
            save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)
            netname = file_name + '.pth'
            d3qn.store_net(netname)
        count += 1'''

    print('finished')
    r, sr, asr, tsr, s, t = performance_test(d3qn[0], 20)
    successrate.append(sr)
    allsuccessrate.append(asr)
    TUserverate.append(tsr)
    stepdone_avg.append(s)
    reward_test_avg.append(r)
    for i in range(env.n_uav):
        netname = file_name+ str(i) + '.pth'
        d3qn[i].store_net(netname)
    save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)


def save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg):
    plt.figure(2)
    #每次迭代的奖励值和
    plt.clf()
    plt.plot(reward_list)
    figname = 'reward' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'reward' + file_name + '.txt'
    save(pathname, reward_list)
    #每次测试每架UAV的平均奖励值
    plt.clf()
    plt.plot(reward_test_avg)
    figname = 'reward_test' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'reward_test' + file_name + '.txt'
    save(pathname, reward_test_avg)
    #单架UAV成功率
    plt.clf()
    plt.plot(successrate)
    figname = 'success_rate' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'success_rate' + file_name + '.txt'
    save(pathname, successrate)
    #三架UAV全成功概率
    plt.clf()
    plt.plot(allsuccessrate)
    figname = 'all_success_rate' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'all_success_rate' + file_name + '.txt'
    save(pathname, allsuccessrate)
    #全部成功的情况下TU服务率
    plt.clf()
    plt.plot(TUserverate)
    figname = 'TU_serve_rate' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'TU_serve_rate' + file_name + '.txt'
    save(pathname, TUserverate)
    #全成功的情况下三架UAV共用步数
    plt.clf()
    plt.plot(stepdone_avg)
    figname = 'stepdone_avg' + file_name + '.png'
    plt.savefig(figname)
    pathname = 'stepdone_avg' + file_name + '.txt'
    save(pathname, stepdone_avg)

def application(pathname):
    d3qn_app = []
    for i in range(env.n_uav):
        d3qn_app.append(D3QN())
        # pathname_localnet = 'localnet_5_5_3_D3_4cen_nTU_obv3.pth'
        pathname_localnet = 'localnet' + pathname + str(i) + '.pth'
        d3qn_app[i].load_net(pathname_localnet)
    #初始化环境参数
    # env.n_uav = 3
    env.n_tus = 15
    env.n_obs = 5
    env.n_ob_dynamic = 5
    env.seed(80)

    state = env.reset(flag_test=True)
    for uav in env.uavs:
        uav.strict_flag = 1
    env.render(1)
    uav_x = []
    uav_y = []
    uav_z = []
    ob_x = []
    ob_y = []
    for _ in range(env.n_uav):
        uav_x.append([])
        uav_y.append([])
        uav_z.append([])
    for _ in range(env.n_ob_dynamic):
        ob_x.append([])
        ob_y.append([])
    while 1:
        action = []
        for i in range(env.n_uav):
            if env.uavs[i].done:
                action.append(-1)
                continue
            action.append(d3qn_app[i].choose_action_apply(state[i]))
        next_state, _, done = env.step(action)

        env.render(1)
        for i in range(env.n_uav):
            if action[i] != -1:
                uav_x[i].append(env.uavs[i].x * env.length)
                uav_y[i].append(env.uavs[i].y * env.width)
                uav_z[i].append(env.uavs[i].H * env.length)
        for i in range(env.n_ob_dynamic):
            ob_x[i].append(env.obs_dynamic[i].x * env.length)
            ob_y[i].append(env.obs_dynamic[i].y * env.width)
        if done.all():
            break
        state = next_state
    env.render(1)
    for i in range(env.n_uav):
        if env.uavs[i].d_tar < (2 * env.uavs[i].stepWay):
            uav_x[i].append(env.uavs[i].x_tar * env.length)
            uav_y[i].append(env.uavs[i].y_tar * env.width)
            uav_z[i].append(env.uavs[i].H * env.length)
        plt.plot(uav_x[i], uav_y[i], 'b-')
    for i in range(env.n_ob_dynamic):
        plt.plot(ob_x[i], ob_y[i], 'r-.')
    env.render_3D(uav_y, uav_x, uav_z)
    plt.pause(5)

def app(pathname):
    for i in range(20):
        application(pathname)

def performance_test(network, num_test, n_UAV=3, n_TU=15, n_dyob = 5, n_ob=5):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
    reward_test = 0
    num_stepdone = 0
    time = 0
    #创建测试环境
    env_test = ENV()
    env_test.n_uav = n_UAV
    env_test.n_tus = n_TU
    env_test.n_ob_dynamic = n_dyob
    env_test.n_obs = n_ob
    #for _ in range(num_test):
    for _ in enumerate(tqdm(range(num_test))):
        state = env_test.reset(flag_test=True)#重置环境
        for uav in env_test.uavs:
            uav.strict_flag = 1
        ep_reward = 0
        flag_allsuccess = True #表示默认情况下认为所有UAV都能成功完成任务
        while 1:
            action = []
            for i in range(env_test.n_uav):
                #这段代码检查UAV是否已完成任务（done标志），如果完成了，将-1添加到动作列表，表示该UAV不再采取行动。否则，将根据代理（network）选择的动作添加到列表中
                if env_test.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(network.choose_action_apply(state[i]))
            next_state, reward, done = env_test.step(action)#执行环境中的动作
            #再次遍历每个UAV，检查UAV是否采取了动作，即action不等于-1
            for i in range(env_test.n_uav):
                if action[i] != -1:
                    ep_reward += reward[i]#如果采取了动作，则添加奖励
            if done:
                break
            state = next_state
        step_done = 0
        step = []
        #遍历环境中的每个UAV，将其完成步数添加到step列表中，并累加到step_done中。
        for uav in env_test.uavs:
            step_done += uav.step_done
            step.append(uav.step_done)
            #检查每个UAV是否成功到达目标
            if uav.d_tar < 2 * uav.stepWay:
                num_success += 1
            else:
                flag_allsuccess = False
        if flag_allsuccess:
            num_allsuccess += 1
            num_stepdone += step_done
            time += max(step)
            for tu in env_test.tus:
                if tu.flag_done:
                    num_TUserved += 1
        reward_test += ep_reward / step_done
    allsuccessrate = num_allsuccess / num_test#全局成功率，所有UAV都成功的比例
    successrate = num_success / num_test / env_test.n_uav#每架UAV的成功率，每架UAV成功的比例
    reward_test = reward_test / num_test#所有测试回合的平均奖励
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus #全局成功情况下的TU服务率
        step_done = num_stepdone / num_allsuccess #平均完成步数
        time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        time_avg = 0

    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg

def performance_test_multiuavs(network, num_test, n_UAV=3, n_TU=15, n_dyob=5, n_ob=5, seed_num = 95):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
    reward_test = 0
    num_stepdone = 0
    collision_cnt = 0
    timeout_cnt = 0
    num_collision=0
    num_uav_collision=0
    time = 0
    env_test = ENV()
    env_test.seed(seed_num)
    env_test.n_uav = n_UAV
    env_test.n_tus = n_TU
    env_test.n_obs = n_ob
    env_test.n_ob_dynamic = n_dyob
    #for _ in range(num_test):
    for _ in enumerate(tqdm(range(num_test))):
        state = env_test.reset(flag_test=True)
        step_cnt = 0
        for i,uav in enumerate(env_test.uavs):
            uav.strict_flag = 1
        ep_reward = 0
        flag_allsuccess = True
        while 1:
            action = []
            for i in range(env_test.n_uav):
                if env_test.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(network[i].choose_action_apply(state[i]))
            next_state, reward, done = env_test.step(action)
            for i in range(env_test.n_uav):
                if action[i] != -1:
                    ep_reward += reward[i]
            if done.all():
                for uav in env_test.uavs:
                    num_collision+=uav.collision_cnt
                    num_uav_collision += uav.uav_collision_cnt
                break
            state = next_state
            step_cnt+=1
        step_done = 0
        collision = False
        time_out = False
        step = []
        for uav in env_test.uavs:
            step_done += uav.step_done
            step.append(uav.step_done)
            if uav.collision:
                collision = True
            if uav.time_out:
                time_out = True
            if uav.d_tar < 2 * uav.stepWay:
                #
                num_success += 1
            else:
                #
                flag_allsuccess = False
        if flag_allsuccess:
            num_allsuccess += 1
            num_stepdone += step_done
            time += max(step)
            for tu in env_test.tus:
                if tu.flag_done:
                    num_TUserved += 1
        elif collision:
            collision_cnt += 1
        elif time_out:
            timeout_cnt += 1
        reward_test += ep_reward / step_done
    allsuccessrate = num_allsuccess / num_test
    successrate = num_success / num_test / env_test.n_uav
    # collision_rate = num_collision / num_test / env_test.n_uav
    collisionsrate = collision_cnt / num_test
    # uav_collision_rate = num_uav_collision / num_test / env_test.n_uav
    reward_test = reward_test / num_test
    timeout_rate = timeout_cnt / num_test
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus
        step_done = num_stepdone / num_allsuccess
        time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        time_avg = 0

    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg, collisionsrate, timeout_rate

if __name__ == '__main__':

    '''for _ in enumerate(tqdm(range(30))):
        plt.pause(60)'''

    train_flag = 0  # 执行训练程序
    application_test = 1  # 执行测试程序
    perf_test = 0  # 执行性能测试程序

    for n in range(3, 4, 1):

        modelname = 'D3QN_24092024_2' + str(n) + 'UAVs_'

        if train_flag:
            # train('_5_5_3_DQN_3cen_nTU_obv3')
            # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
            # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
            train_multiuavs(modelname)

        else:
            if application_test:
                app(modelname)

        if perf_test:
            num_test = 500
            seed_nums = [80, 90, 95]
            test_var = range(3, 10, 1)
            # d3qn_app = D3QN()
            # # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
            # pathname_localnet = 'localnet_5_5_3_DQN_connectivity_3cen_nTU_obv3.pth'
            # d3qn_app.load_net(pathname_localnet)

            d3qn_app = []
            for i in range(n):
                d3qn_app.append(D3QN())
                # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
                pathname_localnet = 'localnet' + modelname + str(i) + '.pth'
                d3qn_app[i].load_net(pathname_localnet)

            UAV_successrate = np.zeros((len(seed_nums), len(test_var)))
            UAV_TUserverate = np.zeros((len(seed_nums), len(test_var)))
            UAV_step_done = np.zeros((len(seed_nums), len(test_var)))
            UAV_time_avg = np.zeros((len(seed_nums), len(test_var)))
            UAV_collision = np.zeros((len(seed_nums), len(test_var)))
            Timeout_rate = np.zeros((len(seed_nums), len(test_var)))

            for i, var in enumerate(test_var):
                for n, seed in enumerate(seed_nums):
                    print('test performance for {} dynamic obstacles'.format(var))
                    _, _, successrate, TUserverate, step_done, avgtime, collision_rate, timeout_rate = performance_test_multiuavs(
                        d3qn_app, num_test, n_UAV=3, n_TU=15, n_dyob=var, n_ob=5, seed_num=seed)
                    UAV_successrate[n, i] = np.array(successrate)
                    UAV_TUserverate[n, i] = np.array(TUserverate)
                    UAV_step_done[n, i] = np.array(step_done)
                    UAV_time_avg[n, i] = np.array(avgtime)
                    UAV_collision[n, i] = np.array(collision_rate)
                    Timeout_rate[n, i] = np.array(timeout_rate)
                print('\nsuccess rate:  %.3f\tTU serve rate:  %.3f\taverage step: %d\taverage time: %d\tcollision rate: %.3f\ttime out rate: %.3f\n'
                      % (np.mean(UAV_successrate, 0)[i], np.mean(UAV_TUserverate, 0)[i],
                        np.mean(UAV_step_done, 0)[i], np.mean(UAV_time_avg, 0)[i],
                        np.mean(UAV_collision, 0)[i], np.mean(Timeout_rate, 0)[i]))

            p = 'A_Result\D3QN_OB_T2_' + str(n) + 'UAVs'

            save(p + '_successrate.txt', np.mean(UAV_successrate, 0).tolist())
            save(p + '_TUserverate.txt', np.mean(UAV_TUserverate, 0).tolist())
            save(p + '_avgstep.txt', np.mean(UAV_step_done, 0).tolist())
            save(p + '_avgtime.txt', np.mean(UAV_time_avg, 0).tolist())
            save(p + '_collision_rate.txt', np.mean(UAV_collision, 0).tolist())
            save(p + '_timeout_rate.txt', np.mean(Timeout_rate, 0).tolist())
