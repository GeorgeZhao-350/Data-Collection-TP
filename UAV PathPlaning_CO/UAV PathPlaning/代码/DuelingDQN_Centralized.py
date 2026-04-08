import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *
from tqdm import tqdm
from save_data import *
import math

env = ENV()
env.reset()

NUM_UAV = env.n_uav
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

class BaseNetWork(nn.Module):
    def __init__(self):
        super(BaseNetWork, self).__init__()

        self.fc1 = nn.Linear(NUM_STATES*NUM_UAV, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        return x

class QNetWork(nn.Module):
    def __init__(self):
        super(QNetWork, self).__init__()

        self.V = nn.Linear(256, 1)
        self.V.weight.data.normal_(0, 0.1)
        self.A = nn.Linear(256, NUM_ACTIONS)
        self.A.weight.data.normal_(0, 0.1)

    def forward(self, state):

        V = self.V(state)
        A = self.A(state)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q

class D3QN():
    def __init__(self):
        self.local_base_net = BaseNetWork()
        self.target_base_net = BaseNetWork()
        self.local_net = []
        self.target_net = []
        for i in range(NUM_UAV):
            self.local_net.append(QNetWork())
            self.target_net.append(QNetWork())

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((NUM_UAV, MEMORY_CAPACITY, NUM_STATES * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.optimizer = []
        for i in range(NUM_UAV):
            self.optimizer.append(torch.optim.Adam(self.local_net[i].parameters(), lr=LEARNING_RATE))
        self.base_optimizer = torch.optim.Adam(self.local_base_net.parameters(), lr=LEARNING_RATE)

    def choose_action_train(self, state, step_done, uav_id):
        #print('state : {}'.format(state))
        state = torch.FloatTensor(state) # get a 1D array
        state_full = torch.cat((state[0, :], state[1, :]))
        for i in range(NUM_UAV - 2):
            state_full = torch.cat((state_full, state[i + 2, :]))
        # print('torch state : {}'.format(state))
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_done / EPS_DECAY)
        #print(eps_threshold)
        if np.random.randn() < eps_threshold:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action_value = self.local_net[uav_id].forward(self.local_base_net.forward(state_full.unsqueeze(0))).detach()
            action = int(torch.max(action_value, 1)[1].data.numpy())
        return action

    def choose_action_apply(self, state, uav_id):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        state_full = torch.cat((state[0, :], state[1, :]))
        for i in range(NUM_UAV - 2):
            state_full = torch.cat((state_full, state[i + 2, :]))
        #将扩展后的状态输入到本地网络（local_net）中进行前向传播，得到每个动作对应的 Q 值。
        action_value = self.local_net[uav_id].forward(self.local_base_net.forward(state_full)).detach()
        action = int(torch.max(action_value, 1)[1].data.numpy())
        return action
    #将一个时间步的状态转换存储到经验回放缓存中。
    def store_transition(self, state, action, reward, next_state):
        index = self.memory_counter % MEMORY_CAPACITY
        for n in range(NUM_UAV):
            if action[n] == -1:
                action[n] = 0
            for i in range(NUM_STATES):
                self.memory[n, index, i] = state[n][i]
            self.memory[n, index, NUM_STATES] = action[n]
            self.memory[n, index, NUM_STATES + 1] = reward[n]
            for i in range(NUM_STATES):
                self.memory[n, index, NUM_STATES + 2 + i] = next_state[n][i]
        self.memory_counter += 1

    def sample(self):
        max_mem=min(self.memory_counter,MEMORY_CAPACITY)
        sample_index = np.random.choice(max_mem, BATCH_SIZE)
        batch_state=torch.zeros((NUM_UAV, BATCH_SIZE, NUM_STATES))
        batch_action = torch.zeros((NUM_UAV, BATCH_SIZE, 1),dtype=torch.int64)
        batch_reward = torch.zeros((NUM_UAV, BATCH_SIZE, 1))
        batch_next_state = torch.zeros((NUM_UAV, BATCH_SIZE, NUM_STATES))
        for i in range(BATCH_SIZE):
            for n in range(NUM_UAV):
                batch_memory = self.memory[n, sample_index[i], :]
                batch_state[n, i, :] = torch.FloatTensor(batch_memory[ :NUM_STATES])
                batch_action[n, i] = torch.LongTensor(batch_memory[ NUM_STATES:NUM_STATES + 1].astype(int))
                batch_reward[n, i] = torch.FloatTensor(batch_memory[ NUM_STATES + 1:NUM_STATES + 2])
                batch_next_state[n, i, :] = torch.FloatTensor(batch_memory[ -NUM_STATES:])

        return sample_index, batch_state, batch_action, batch_reward, batch_next_state

    def learn(self):
        #统一网络参数
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            for i in range(NUM_UAV):
                self.target_net[i].load_state_dict(self.local_net[i].state_dict())
        self.learn_step_counter += 1
        sample_index, batch_state, batch_action, batch_reward, batch_next_state = self.sample()

        batch_state_full =  torch.cat((batch_state[0, :, :], batch_state[1, :, :]), 1)
        for i in range(NUM_UAV - 2):
            batch_state_full = torch.cat((batch_state_full, batch_state[i + 2, :, :]))
        batch_next_state_full = torch.cat((batch_next_state[0, :, :], batch_next_state[1, :, :]), 1)
        for i in range(NUM_UAV - 2):
            batch_next_state_full = torch.cat((batch_next_state_full, batch_next_state[i + 2, :, :]))

        q_local = []
        for n in range(NUM_UAV):
            q_local.append(self.local_net[n](self.local_base_net.forward(batch_state_full)).gather(1, batch_action[n]))

        #DQN
        #q_next = self.target_net(batch_next_state).detach()

        #DDQN
        q_target = []
        for n in range(NUM_UAV):
            # local_action_value = self.local_net[n].forward(self.local_base_net.forward(batch_state_full)).detach()
            # local_action = torch.max(local_action_value, 1)[1].data.numpy()
            # local_action_t = []
            # for i in range(BATCH_SIZE):
            #     local_action_t.append([local_action[i]])
            # local_action_t = np.array(local_action_t, ndmin=2)[:, :]
            # local_action = torch.LongTensor(local_action_t.astype(int))
            q_next = self.target_net[n].forward(self.local_base_net.forward(batch_next_state_full)).detach().max(1)[0]
            q_target.append(batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1))

        loss = 0
        for n in range(NUM_UAV):
            loss += self.loss_func(q_local[n], q_target[n])

        for n in range(NUM_UAV):
            self.optimizer[n].zero_grad()
        self.base_optimizer.zero_grad()
        loss.backward()
        for n in range(NUM_UAV):
            self.optimizer[n].step()
        self.base_optimizer.step()

    def store_net(self, pathname):
        for n in range(NUM_UAV):
            torch.save(self.local_net[n].state_dict(), 'localnet' + pathname + 'agent' + str(n) + '.pth')
        #torch.save(self.target_net.state_dict(), 'targetnet' + pathname)
        torch.save(self.local_base_net.state_dict(), 'localnet' + pathname + 'base' + '.pth')

    def load_net(self, pathname_loaclnet):
        for n in range(NUM_UAV):
            self.local_net[n].load_state_dict(torch.load(pathname_loaclnet + 'agent' + str(n) + '.pth'))
            self.target_net[n].load_state_dict(self.local_net[n].state_dict())
        self.local_base_net.load_state_dict(torch.load(pathname_loaclnet + 'base' + '.pth'))
        self.target_base_net.load_state_dict(self.local_base_net.state_dict())


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
    netname = file_name
    d3qn.store_net(netname)
    save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)

def train_multiuavs(file_name):
    d3qn = D3QN()
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
        state, _ = env.reset()

        ep_reward = np.zeros(shape=(env.n_uav))
        while 1:
            action = []
            for i in range(env.n_uav):
                if env.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(d3qn.choose_action_train(state, step, i))

            if d3qn.memory_counter >= TRAIN_THRESHOLD and step%10 == 0:
                d3qn.learn()
            step += 1
            next_state, reward, done, _ = env.step(action)
            d3qn.store_transition(state, action, reward, next_state)
            for i in range(env.n_uav):
                ep_reward[i] += reward[i]

            if done.all():
                break
            state = next_state

        reward_list.append(ep_reward) #三个智能体各自的reward

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
    # r, sr, asr, tsr, s, t = performance_test(d3qn[0], 20)
    # successrate.append(sr)
    # allsuccessrate.append(asr)
    # TUserverate.append(tsr)
    # stepdone_avg.append(s)
    # reward_test_avg.append(r)
    netname = file_name + '.pth'
    d3qn.store_net(netname)
    print('finished')
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

def application(pathname_localnet):
    d3qn_app = D3QN()
    d3qn_app.load_net(pathname_localnet)
    #初始化环境参数
    state, _ = env.reset(flag_test=True)
    for uav in env.uavs:
        uav.strict_flag = 1
    env.render(1)
    uav_x = []
    uav_y = []
    ob_x = []
    ob_y = []
    for _ in range(env.n_uav):
        uav_x.append([])
        uav_y.append([])
    for _ in range(env.n_ob_dynamic):
        ob_x.append([])
        ob_y.append([])
    while 1:
        #在每次迭代中选择无人机的动作，执行环境步骤，更新状态，并记录无人机和障碍物的轨迹数据，当某个终止条件（done）满足时，退出循环
        action = []
        for i in range(env.n_uav):
            if env.uavs[i].done:
                action.append(-1)
                continue
            action.append(d3qn_app.choose_action_apply(state, i))
        next_state, _, done, _ = env.step(action)

        env.render(1)
        #更新无人机的位置
        for i in range(env.n_uav):
            if action[i] != -1:
                uav_x[i].append(env.uavs[i].x * env.length)
                uav_y[i].append(env.uavs[i].y * env.width)
        #更新动态障碍物位置
        for i in range(env.n_ob_dynamic):
            ob_x[i].append(env.obs_dynamic[i].x * env.length)
            ob_y[i].append(env.obs_dynamic[i].y * env.width)
        if done:
            break
        state = next_state
    env.render(1)
    for i in range(env.n_uav):
        if env.uavs[i].d_tar < (2*env.uavs[i].stepWay):
            uav_x[i].append(env.uavs[i].x_tar * env.length)
            uav_y[i].append(env.uavs[i].y_tar * env.width)
        plt.plot(uav_x[i], uav_y[i], 'b-')
    for i in range(env.n_ob_dynamic):
        plt.plot(ob_x[i], ob_y[i], 'r-.')
    plt.pause(5)

def app(pathname_localnet):
    random.seed(10)
    for i in range(20):
        application(pathname_localnet)

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
        state, _ = env_test.reset(flag_test=True)#重置环境
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
            next_state, reward, done, _ = env_test.step(action)#执行环境中的动作
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

def performance_test_multiuavs(network, num_test, n_UAV=3, n_TU=15, n_dyob=5, n_ob=5):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
    num_collision = 0
    reward_test = 0
    num_stepdone = 0
    time = 0
    env_test = ENV()
    env_test.n_uav = n_UAV
    env_test.n_tus = n_TU
    env_test.n_obs = n_ob
    env_test.n_ob_dynamic = n_dyob
    #for _ in range(num_test):
    for _ in enumerate(tqdm(range(num_test))):
        state, _ = env_test.reset(flag_test=True)
        for uav in env_test.uavs:
            uav.strict_flag = 1
        ep_reward = 0
        flag_allsuccess = True
        while 1:
            action = []
            for i in range(env_test.n_uav):
                if env_test.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(network.choose_action_apply(state, i))
            next_state, reward, done, _ = env_test.step(action)
            for i in range(env_test.n_uav):
                if action[i] != -1:
                    ep_reward += reward[i]
            if done.all():
                for uav in env_test.uavs:
                    num_collision+=uav.collision_cnt
                break
            state = next_state
        step_done = 0
        step = []
        for uav in env_test.uavs:
            step_done += uav.step_done
            step.append(uav.step_done)
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
        reward_test += ep_reward / step_done
    allsuccessrate = num_allsuccess / num_test
    successrate = num_success / num_test / env_test.n_uav
    collision_rate = num_collision / num_test / env_test.n_uav
    reward_test = reward_test / num_test
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus
        step_done = num_stepdone / num_allsuccess
        time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        time_avg = 0


    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg, collision_rate

if __name__ == '__main__':

    '''for _ in enumerate(tqdm(range(30))):
        plt.pause(60)'''

    train_flag = 1  # 执行训练程序
    application_test = 0  # 执行测试程序
    perf_test = 1  # 执行性能测试程序

    filename = 'DuelingDQN_Centralized'

    if train_flag:
        # train('_5_5_3_DQN_3cen_nTU_obv3')
        # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
        # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
        train_multiuavs(filename)


    else:
        if application_test:
            app(filename)

    if perf_test:
        num_test = 100
        # d3qn_app = D3QN()
        # # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
        # pathname_localnet = 'localnet_5_5_3_DQN_connectivity_3cen_nTU_obv3.pth'
        # d3qn_app.load_net(pathname_localnet)

        d3qn_app= D3QN()
        # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
        pathname_localnet = 'localnet' + filename
        d3qn_app.load_net(pathname_localnet)

        UAV_successrate = []
        UAV_TUserverate = []
        UAV_step_done = []
        UAV_time_avg = []
        UAV_collision = []

        for i in range(9, 30, 3):
            print('test performance for {} dynamic obstacles'.format(i))
            _, _, successrate, TUserverate, step_done, avgtime, collision_rate = performance_test_multiuavs(d3qn_app, num_test,
                                                                                              n_TU=i, n_dyob=5, n_ob=5, n_UAV=NUM_UAV)
            UAV_successrate.append(successrate)
            UAV_TUserverate.append(TUserverate)
            UAV_step_done.append(step_done)
            UAV_time_avg.append(avgtime)
            UAV_collision.append(collision_rate)
            print('\nsuccess rate:  %.3f\tTU serve rate:  %.3f\taverage step: %d\taverage time: %d\tCollision rate:  %.3f\n' % (
            successrate, TUserverate, step_done, avgtime, collision_rate))

        p = 'DuelingDQN_Centralized_07012025_1_T1_TU'

        save(p + '_successrate.txt', UAV_successrate)
        save(p + '_TUserverate.txt', UAV_TUserverate)
        save(p + '_avgstep.txt', UAV_step_done)
        save(p + '_avgtime.txt', UAV_time_avg)
        save(p + '_collision_rate.txt', UAV_collision)
