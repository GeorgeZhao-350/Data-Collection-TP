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
GAMMA = 0.95
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
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, 256)
        self.fc3.weight.data.normal_(0, 0.1)
        #self.fc4 = nn.Linear(256, 256)
        #self.fc4.weight.data.normal_(0, 0.1)

        self.V = nn.Linear(256, 1)
        self.V.weight.data.normal_(0, 0.1)
        self.A = nn.Linear(256, NUM_ACTIONS)
        self.A.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))

        V = self.V(x)
        A = self.A(x)
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
        #print('torch state : {}'.format(state))
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
        action_value = self.local_net.forward(state).detach()
        action = int(torch.max(action_value, 1)[1].data.numpy())
        return action

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
    episodes = 5000
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

                if d3qn.memory_counter >= TRAIN_THRESHOLD and step%10 == 0:
                    d3qn.learn()
                step += 1
            next_state, reward, done = env.step(action)
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

def application():
    d3qn_app = D3QN()
    pathname_localnet = 'localnetd3qn_26112025.pth'
    d3qn_app.load_net(pathname_localnet)
    env.n_obs = 10
    env.n_ob_dynamic = 10
    env.n_tus = 50
    env.seed(96)
    state = env.reset(flag_test=True)
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
        action = []
        for i in range(env.n_uav):
            if env.uavs[i].done:
                action.append(-1)
                continue
            action.append(d3qn_app.choose_action_apply(state[i]))
        next_state, _, done = env.step(action)

        env.render(1)
        for i in range(env.n_uav):
            if action[i] != -1:
                uav_x[i].append(env.uavs[i].x * env.length)
                uav_y[i].append(env.uavs[i].y * env.width)
        for i in range(env.n_ob_dynamic):
            ob_x[i].append(env.obs_dynamic[i].x * env.length)
            ob_y[i].append(env.obs_dynamic[i].y * env.width)
        if done.all():
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

def app():
    # random.seed(10)
    for i in range(20):
        application()

def performance_test(network, num_test, n_UAV=3, n_TU=15, n_dyob = 5, n_ob=5):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
    reward_test = 0
    num_stepdone = 0
    collision_cnt = 0
    timeout_cnt = 0
    time = 0
    env_test = ENV()
    env_test.n_uav = n_UAV
    env_test.n_tus = n_TU
    env_test.n_ob_dynamic = n_dyob
    env_test.n_obs = n_ob
    #for _ in range(num_test):
    for _ in enumerate(tqdm(range(num_test))):
        state = env_test.reset(flag_test=True)
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
                action.append(network.choose_action_apply(state[i]))
            next_state, reward, done = env_test.step(action)
            for i in range(env_test.n_uav):
                if action[i] != -1:
                    ep_reward += reward[i]
            if done.all():
                break
            state = next_state
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
                num_success += 1
            else:
                flag_allsuccess = False
        if flag_allsuccess:
            num_allsuccess += 1
            num_stepdone += step_done
            # time += max(step)
            for tu in env_test.tus:
                if tu.flag_done:
                    num_TUserved += 1
        elif collision:
            collision_cnt += 1
            # time += max(step)
        elif time_out:
            timeout_cnt += 1
            # time += max(step)
        time += max(step)
        reward_test += ep_reward / step_done
    allsuccessrate = num_allsuccess / num_test
    successrate = num_success / num_test / env_test.n_uav
    reward_test = reward_test / num_test
    collisionsrate = collision_cnt / num_test
    timeout_rate = timeout_cnt / num_test
    time_avg = time / num_test
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus
        step_done = num_stepdone / num_allsuccess
        # time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        # time_avg = 0

    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg, collisionsrate, timeout_rate

if __name__ == '__main__':

    '''for _ in enumerate(tqdm(range(30))):
        plt.pause(60)'''

    '''train_flag = 1
    if train_flag:
        train('_5_5_3_D3_cen_nTU_obv3')
    else:
        app()'''
    # app()

    num_test = 500
    seed_nums = [80, 90, 95]
    test_var = range(10, 11, 1)
    d3qn_app = D3QN()
    pathname_localnet = 'localnetd3qn_26112025.pth'
    d3qn_app.load_net(pathname_localnet)

    UAV_successrate = np.zeros((len(seed_nums), len(test_var)))
    UAV_TUserverate = np.zeros((len(seed_nums), len(test_var)))
    UAV_step_done = np.zeros((len(seed_nums), len(test_var)))
    UAV_time_avg = np.zeros((len(seed_nums), len(test_var)))
    UAV_collision = np.zeros((len(seed_nums), len(test_var)))
    Timeout_rate = np.zeros((len(seed_nums), len(test_var)))

    for i, var in enumerate(test_var):
        for n, seed in enumerate(seed_nums):
            print('test performance for {} uavs, {} dynamic obstacles'.format(3, i))
            _, _, successrate, TUserverate, step_done, avgtime, collision_rate, timeout_rate = performance_test(d3qn_app, num_test, n_TU=50, n_dyob=10, n_ob=10)
            UAV_successrate[n, i] = np.array(successrate)
            UAV_TUserverate[n, i] = np.array(TUserverate)
            UAV_step_done[n, i] = np.array(step_done)
            UAV_time_avg[n, i] = np.array(avgtime)
            UAV_collision[n, i] = np.array(collision_rate)
            Timeout_rate[n, i] = np.array(timeout_rate)
        print(
            '\nsuccess rate:  %.3f\tTU serve rate:  %.3f\taverage step: %d\taverage time: %d\tcollision rate: %.3f\ttime out rate: %.3f\n'
            % (np.mean(UAV_successrate, 0)[i], np.mean(UAV_TUserverate, 0)[i],
               np.mean(UAV_step_done, 0)[i], np.mean(UAV_time_avg, 0)[i],
               np.mean(UAV_collision, 0)[i], np.mean(Timeout_rate, 0)[i]))
        i += 1
    p = 'd3qn_testing_OB10_DOB10_TU50_Small_Large'

    save(p + '_successrate.txt', np.mean(UAV_successrate, 0).tolist())
    save(p + '_TUserverate.txt', np.mean(UAV_TUserverate, 0).tolist())
    save(p + '_avgstep.txt', np.mean(UAV_step_done, 0).tolist())
    save(p + '_avgtime.txt', np.mean(UAV_time_avg, 0).tolist())
    save(p + '_collision_rate.txt', np.mean(UAV_collision, 0).tolist())
    save(p + '_timeout_rate.txt', np.mean(Timeout_rate, 0).tolist())
