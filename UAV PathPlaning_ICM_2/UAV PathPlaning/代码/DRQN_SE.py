import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *
from tqdm import tqdm
from save_data import *
import math
import random

env = ENV()
env.reset()

NUM_STATES = 2 + 3*env.num_tu_can + 2*2 + env.num_lidar #+ 3*3
NUM_ACTIONS = env.num_actions
LEARNING_RATE = 0.0001     #学习率 # 0.0001
GAMMA = 0.95               # default 0.95
MEMORY_CAPACITY = 80000     #经验池容量0.830
TRAIN_THRESHOLD = 30000     #训练开始阈值
BATCH_SIZE = 16    #单次学习取记录数 # default 128
EPS_START = 0.8
EPS_END = 0.1
EPS_DECAY = 5000
Q_NETWORK_ITERATION = 100    #学习多少次后统一两个网络的参数 # defult 100

BETA_1=0.9
BETA_2=0.9
U_BOUNDARY=0.5

class DRQNetWork(nn.Module):
    def __init__(self):
        super(DRQNetWork, self).__init__()
        self.LSTM_hidsize = 128

        self.fc1 = nn.Linear(NUM_STATES, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, 256)
        self.fc3.weight.data.normal_(0, 0.1)
        # self.fc4 = nn.Linear(256, 256)
        # self.fc4.weight.data.normal_(0, 0.1)
        # self.conv1=nn.Conv1d(1,1,3,1)
        # self.conv1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(self.LSTM_hidsize, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

        self.h_unbatch=torch.zeros(1,self.LSTM_hidsize)
        self.c_unbatch=torch.zeros(1,self.LSTM_hidsize)
        self.rnn=nn.LSTM(256,self.LSTM_hidsize)
        self.h_minibatch = torch.zeros(BATCH_SIZE, self.LSTM_hidsize)
        self.c_minibatch = torch.zeros(BATCH_SIZE, self.LSTM_hidsize)

    def forward(self, state):
        # state=torch.reshape(state, (state.shape[0], 1, state.shape[1]))
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        shape=x.size()
        x=torch.reshape(x,[1,shape[1]])
        x,(self.h_unbatch,self.c_unbatch) = self.rnn(x,(self.h_unbatch,self.c_unbatch))
        #x = torch.relu(self.fc4(x))

        out = self.out(x)
        #out = torch.tanh(self.out(x))

        return torch.reshape(out,(1,-1))

    def clear_memorycell(self):
        self.h_unbatch = torch.zeros(1, self.LSTM_hidsize)
        self.c_unbatch = torch.zeros(1, self.LSTM_hidsize)

    def batch_forward(self, batch_state):
        self.h_minibatch=torch.zeros(1,BATCH_SIZE,self.LSTM_hidsize)
        self.c_minibatch=torch.zeros(1,BATCH_SIZE,self.LSTM_hidsize)

        # state=torch.reshape(state, (state.shape[0], 1, state.shape[1]))
        x = torch.relu(self.fc1(batch_state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        shape = x.size()
        x = torch.reshape(x, [1, shape[0], shape[1]])
        x,(self.h_minibatch,self.c_minibatch) = self.rnn(x,(self.h_minibatch,self.c_minibatch))
        #x = torch.relu(self.fc4(x))

        out = self.out(x)
        #out = torch.tanh(self.out(x))

        return torch.reshape(out,(BATCH_SIZE,-1))


class Replay_Memory:
    def __init__(self):
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2+1))
        self.priorities = np.zeros(MEMORY_CAPACITY)
        self.td_err = np.zeros((MEMORY_CAPACITY, 1))
        self.beta = 0.4

    def store_transition(self, state, action, reward, next_state, terminal):
        index = self.memory_counter % MEMORY_CAPACITY
        for i in range(NUM_STATES):
            self.memory[index, i] = state[i]
        self.memory[index, NUM_STATES] = action
        self.memory[index, NUM_STATES + 1] = reward
        for i in range(NUM_STATES):
            self.memory[index, NUM_STATES + 2 + i] = next_state[i]
        self.memory[index, 2*NUM_STATES + 2] = terminal
        # self.td_err[index]=td_err
        # self.prop_priority(index,td_err)
        self.memory_counter += 1
        return self.memory[index]


    def scaled_prob(self):
        # probability updates
        P = np.array(self.priorities, dtype=np.float32)
        if P.sum() != 0:
            P /= P.sum()
        P += (1 - P.sum()) / min(self.memory_counter, MEMORY_CAPACITY)
        return abs(P)

    def prob_imp(self, prob, beta):
        # return importance
        self.beta = beta
        max_mem = min(self.memory_counter, MEMORY_CAPACITY)
        i = (1 / max_mem * 1 / prob) ** (-self.beta)
        i /= max(i)
        return i

    def sample(self):
        # find the batch and importance using proportional prioritization
        self.beta = np.min([1., 0.001 + self.beta])

        max_mem = min(self.memory_counter, MEMORY_CAPACITY)
        # probability = self.scaled_prob()
        index = np.random.choice(np.arange(max_mem), BATCH_SIZE, replace=False)
        # imp = self.prob_imp(probability[index], self.beta)
        batch_memory = self.memory[index, :]
        return batch_memory, index

    def prop_priority(self, i, err, c=0.001, alpha_value=-0.015):  # Alpha_val orig=0.7, c=1.1
        # proportional prioritization
        self.priorities[i] = (np.abs(err) + c) ** alpha_value

    def clear_priority(self, i):  # Alpha_val orig=0.7, c=1.1
        # proportional prioritization
        self.priorities[i] = 0.0001


class Optimizer():
    def __init__(self, network_params):
        self.momentum=[]
        self.variance=[]
        self.momentum_norm = []
        self.variance_norm = []
        self.delta_t=[]
        self.step_cnt=0
        self.network_params=network_params
        for i, _ in enumerate(self.network_params):
            self.momentum.append(torch.zeros((1)))
            self.variance.append(torch.zeros((1)))
            self.momentum_norm.append(torch.zeros((1)))
            self.variance_norm.append(torch.zeros((1)))
            self.delta_t.append(torch.zeros((1)))

    def step(self):
        self.step_cnt+=1
        eta=random.uniform(0,U_BOUNDARY)
        for i, param in enumerate(self.network_params):
            self.momentum[i]=BETA_1*self.momentum[i]+(1-BETA_1)*param.grad
            self.variance[i]=BETA_2*self.variance[i]+(1-BETA_2)*(param.grad)**2
            self.momentum_norm[i]=self.momentum[i]/(1-BETA_1**self.step_cnt)
            self.variance_norm[i] = self.variance[i] / (1 - BETA_2 ** self.step_cnt)
            sqrt=torch.sqrt((1-eta)*(self.momentum_norm[i]**2)+eta*self.variance_norm[i])
            self.delta_t[i]=torch.abs(self.momentum_norm[i])*self.momentum[i]/(sqrt+(1e-8))
            param-=LEARNING_RATE*self.delta_t[i]


class DRQN():
    def __init__(self, shared_repbuffer):
        self.local_net = DRQNetWork()
        self.target_net = DRQNetWork()

        self.learn_step_counter = 0
        self.replay_buffer = shared_repbuffer

        self.loss_func = nn.MSELoss(reduction="mean")  # mean square
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
            with torch.no_grad():
                action_value = self.local_net.forward(state).detach()
            action_value=torch.reshape(action_value, (1, -1))
            action = int(torch.max(action_value, 1)[1].data.numpy())
        return action

    def choose_action_apply(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        with torch.no_grad():
            action_value = self.local_net.forward(state).detach()
        action_value = torch.reshape(action_value, (1, -1))
        action = int(torch.max(action_value, 1)[1].data.numpy())
        return action

    def tderr_calc(self, state, action, reward, next_state, terminal): # size of menory array: (1,-1)
        batch_state = torch.reshape(torch.FloatTensor(state),(1,-1))
        batch_action = torch.reshape(torch.LongTensor(np.array(action,dtype=np.float32)),(1,-1))
        batch_reward = torch.reshape(torch.FloatTensor(np.array(reward,dtype=np.float32)),(1,-1))
        batch_next_state = torch.reshape(torch.FloatTensor(next_state),(1,-1))
        batch_terminal=torch.reshape(torch.logical_not(torch.BoolTensor(np.array(terminal,dtype=bool))),(1,-1))
        q_local = self.local_net.forward(batch_state).gather(1, batch_action)

        # DQN
        # q_next = self.target_net(batch_next_state).detach()

        # DDQN
        local_action_value = self.local_net.forward(batch_next_state).detach()
        local_action = torch.max(local_action_value, 1)[1].data.numpy()
        local_action = torch.reshape(torch.LongTensor(local_action.astype(int)),(1,-1))
        q_next = self.target_net.forward(batch_next_state).gather(1, local_action).detach()

        # 这里有问题，如果next state为中止状态，q值应为batch_reward?
        q_target = batch_reward + GAMMA * q_next.max(1)[0]*(batch_terminal.to(dtype=int))
        td_err = q_target - q_local
        return td_err

    def learn(self):
        #统一网络参数
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.local_net.state_dict())  #update target
        self.learn_step_counter += 1

        #在[0,MEMORY_CAPACITY)内输出BATCH_SIZE个数字
        batch_memory, sample_index=self.replay_buffer.sample()
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_terminal = torch.reshape(torch.logical_not(torch.BoolTensor(batch_memory[:, 2 * NUM_STATES + 2])),
                                       (BATCH_SIZE, -1))
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])
        q_local = self.local_net.batch_forward(batch_state).gather(1, batch_action)

        #DQN
        q_next = self.target_net.batch_forward(batch_next_state).detach()

        #DDQN
        '''local_action_value = self.local_net.forward(batch_next_state).detach()
        local_action = torch.max(local_action_value, 1)[1].data.numpy()
        local_action_t = []
        for i in range(BATCH_SIZE):
            local_action_t.append([local_action[i]])
        local_action_t = np.array(local_action_t, ndmin=2)[:, :]
        local_action = torch.LongTensor(local_action_t.astype(int))
        q_next = self.target_net(batch_next_state).gather(1, local_action).detach()'''

        #这里有问题，如果next state为中止状态，q值应为batch_reward?
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)*(batch_terminal.to(dtype=int))

        for i in range(BATCH_SIZE):
            idx = sample_index[i]
            self.replay_buffer.clear_priority(idx)

        loss = self.loss_func(q_local, q_target)
        self.local_net.zero_grad()
        loss.backward()
        with torch.no_grad():
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
    exp_memory=Replay_Memory()
    d3qn = DRQN(exp_memory)
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
        d3qn.local_net.clear_memorycell()
        ep_reward = 0
        while 1:
            action = []
            for i in range(env.n_uav):
                if env.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(d3qn.choose_action_train(state[i], step))

                if exp_memory.memory_counter >= TRAIN_THRESHOLD and step%1 == 0:
                    d3qn.learn()
                step += 1
            next_state, reward, done = env.step(action)
            for i in range(env.n_uav):
                if action[i] != -1:
                    exp_memory.store_transition(state[i], action[i], reward[i], next_state[i], done)
                    ep_reward += reward[i]

            if done.all():
                break
            state = next_state

        reward_list.append(ep_reward)

        if exp_memory.memory_counter < TRAIN_THRESHOLD:
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
    r, sr, asr, tsr, s, t = performance_test(d3qn, 20)
    successrate.append(sr)
    allsuccessrate.append(asr)
    TUserverate.append(tsr)
    stepdone_avg.append(s)
    reward_test_avg.append(r)
    netname = file_name + '.pth'
    d3qn.store_net(netname)
    save_fig(file_name, reward_list, reward_test_avg, successrate, TUserverate, allsuccessrate, stepdone_avg)

def train_distributed(file_name):
    exp_memory = Replay_Memory()
    d3qn = []
    for _ in range(env.n_uav):
        d3qn.append(DRQN(exp_memory))
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
        for i in range(env.n_uav):
            d3qn[i].local_net.clear_memorycell()
        ep_reward = np.zeros(shape=(env.n_uav))
        while 1:
            action = []
            for i in range(env.n_uav):
                if env.uavs[i].done:
                    action.append(-1)
                    continue
                action.append(d3qn[i].choose_action_train(state[i], step))
                if exp_memory.memory_counter >= TRAIN_THRESHOLD and step%10 == 0:
                    d3qn[i].learn()
            step += 1
            next_state, reward, done = env.step(action)
            # td_err = np.zeros(3, dtype=np.float32)
            for i in range(env.n_uav):
                if action[i] != -1:
                    # td_err[i] = d3qn[i].tderr_calc(state[i], action[i], reward[i], next_state[i], done[i])
                    exp_memory.store_transition(state[i], action[i], reward[i], next_state[i], done[i])
                    ep_reward[i] += reward[i]

            if done.all():
                break
            state = next_state

        reward_list.append(ep_reward) #三个智能体各自的reward

        if exp_memory.memory_counter < TRAIN_THRESHOLD:
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

def application():
    exp_memory = Replay_Memory()
    d3qn_app = DRQN(exp_memory)
    # pathname_localnet = 'localnet_5_5_3_DQN_nTU_obv3.pth'
    pathname_localnet = 'localnet_5_5_3_DQN_connectivity_3cen_nTU_obv3.pth'
    # pathname_localnet = 'localnet_5_5_3_Dqn_nTU_obv3.pth'


    d3qn_app.load_net(pathname_localnet)

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

def app():
    random.seed(10)
    for i in range(2):
        application()

def performance_test(network, num_test, n_UAV=3, n_TU=15, n_dyob=5, n_ob=5):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
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
        state = env_test.reset(flag_test=True)
        network.local_net.clear_memorycell()
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
    reward_test = reward_test / num_test
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus
        step_done = num_stepdone / num_allsuccess
        time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        time_avg = 0


    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg

def performance_test_distributed(network, num_test, n_UAV=3, n_TU=15, n_dyob=5, n_ob=5):
    num_success = 0
    num_allsuccess = 0
    num_TUserved = 0
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
        state = env_test.reset(flag_test=True)
        for i, uav in enumerate(env_test.uavs):
            uav.strict_flag = 1
            network[i].local_net.clear_memorycell()
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
    reward_test = reward_test / num_test
    if num_allsuccess > 0:
        TUserverate = num_TUserved / num_allsuccess / env_test.n_tus
        step_done = num_stepdone / num_allsuccess
        time_avg = time / num_allsuccess
    else:
        TUserverate = 0
        step_done = 0
        time_avg = 0


    return reward_test, successrate, allsuccessrate, TUserverate, step_done, time_avg

if __name__ == '__main__':

    '''for _ in enumerate(tqdm(range(60))):
        plt.pause(60)'''

    train_flag = 1
    application_test=0
    perf_test=1

    if train_flag:
        # train('_5_5_3_DQN_3cen_nTU_obv3')
        # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
        # train('_5_5_3_DQN_connectivity_3cen_nTU_obv3')
        train_distributed('DRQN_RandomSerial')


    else:
        if application_test:
            app()

    if perf_test:
        num_test = 1000
        # d3qn_app = D3QN()
        # # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
        # pathname_localnet = 'localnet_5_5_3_DQN_connectivity_3cen_nTU_obv3.pth'
        # d3qn_app.load_net(pathname_localnet)
        d3qn_app=[]
        for i in range(3):
            d3qn_app.append(DRQN(np.zeros(1, dtype=np.float32)))
            # pathname_localnet = 'localnet_5_5_3_DQN_3cen_nTU_obv3.pth'
            pathname_localnet = 'localnetDRQN_RandomSerial'+ str(i) + '.pth'
            d3qn_app[i].load_net(pathname_localnet)

        UAV_successrate = []
        UAV_TUserverate = []
        UAV_step_done = []
        UAV_time_avg = []

        for i in range(3, 8):
            print('test performance for {} dynamic obstacles'.format(i))
            _, _, successrate, TUserverate, step_done, avgtime = performance_test_distributed(d3qn_app, num_test, n_TU=15, n_dyob=i)
            UAV_successrate.append(successrate)
            UAV_TUserverate.append(TUserverate)
            UAV_step_done.append(step_done)
            UAV_time_avg.append(avgtime)
            print('\nsuccess rate:  %.3f\tTU serve rate:  %.3f\taverage step: %d\taverage time: %d\n' % (successrate, TUserverate, step_done, avgtime))

        p = 'DRQN_RandomSerial'

        save(p + '_successrate.txt', UAV_successrate)
        save(p + '_TUserverate.txt', UAV_TUserverate)
        save(p + '_avgstep.txt', UAV_step_done)
        save(p + '_avgtime.txt', UAV_time_avg)