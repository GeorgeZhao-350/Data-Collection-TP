import numpy as np
import random
import math
import matplotlib.pyplot as plt
from UAV import UAV
#from UAV import UAV_biginput
from TerminalUser import TerminalUser
from obstacle import obstacle
from obstacle import obstacle_dynamic

def fix_theta(theta, down, up):
    t = theta
    d = up - down
    t += d if t < down else 0
    t -= d if t > up else 0
    return t
class ENV():
    def __init__(self):
        self.length = 1000
        self.width = 1000
        self.n_uav = 3  #目标位置的数量
        self.n_tus = 15 #终端用户的数量
        self.n_obs = 5 #固定障碍物的数量
        self.n_ob_dynamic = 5 #移动障碍物的数量
        self.num_actions = 10 + 1
        self.num_tu_can = 5
        self.tu_allocation=[]
        self.tu_service_state = []
        self.maxI = 5
        self.num_lidar = 16
        self.maxr = 0.07
        self.target = []
        self.tus = []
        self.obs = [] #创建的障碍物对象
        self.obs_dynamic = []
        self.theta_ob_avg = 0     #移动障碍物平均角速度
        self.uavs = []
        self.hover_flag = []
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def seed(self,seed = None):
        if seed:
            if seed >= 0:
                np.random.seed(seed)
    #重置仿真环境
    def reset(self, flag_test = False):
        flag_vertical = True#round(random.random())
        flag_up_right = True#round(random.random())

        #固定障碍物
        n_obs = self.n_obs if flag_test else np.random.randint(3,11) #固定障碍物的数量
        self.obs = []
        for _ in range(n_obs): #在随机的位置 (ob_x, ob_y) 创建固定障碍物
            while 1:
                ob_x = random.uniform(0.15, 0.85) #障碍物的位置必须在 (0.15, 0.85) 的范围内
                ob_y = random.uniform(0.15, 0.85)
                flag = True
                '''if math.sqrt((ob_x - self.target[0]) ** 2 + (ob_y - self.target[1]) ** 2) < 0.2:
                    continue'''
                for ob in self.obs: #障碍物之间的最小距离必须大于 0.14
                    if math.sqrt((ob_x - ob.x) ** 2 + (ob_y - ob.y) ** 2) < 0.14:
                        flag = False
                        break
                if flag:
                    break
            ob_r = random.uniform(0.04, self.maxr) #障碍物的半径
            self.obs.append(obstacle(ob_x, ob_y, ob_r))

        #终端用户
        n_tus = self.n_tus if flag_test else np.random.randint(10, 26)
        self.tus = []
        for i in range(n_tus):#在随机的位置 (tu_x, tu_y) 创建终端用户
            while 1:
                if flag_vertical:
                    tu_x = random.uniform(0, 1)
                    tu_y = random.uniform(0.2, 0.8)
                else:
                    tu_x = random.uniform(0.2, 0.8)
                    tu_y = random.uniform(0, 1)
                flag = True
                #终端用户与固定障碍物之间的最小距离必须大于 0.1
                for ob in self.obs:
                    if math.sqrt((tu_x - ob.x) ** 2 + (tu_y - ob.y) ** 2) < 0.1:
                        flag = False
                        break
                if flag == False:
                    continue
                #终端用户之间的最小距离必须大于 0.08
                for tu in self.tus:
                    if math.sqrt((tu_x - tu.x) ** 2 + (tu_y - tu.y) ** 2) < 0.08:
                        flag = False
                        break
                if flag:
                    break
            I = np.random.randint(3, self.maxI+1) #为每个终端用户生成一个随机的 I 值（数据量）
            self.tus.append(TerminalUser(i, tu_x, tu_y, I))

        # 移动障碍物
        n_ob_dy = self.n_ob_dynamic if flag_test else np.random.randint(3, 8)
        self.obs_dynamic = []
        theta_temp = 0 #累积移动障碍物的角速度
        for i in range(n_ob_dy):
            while (1):
                ob_x = random.uniform(0.15, 0.85)
                ob_y = random.uniform(0.15, 0.85)
                flag = True
                for ob_d in self.obs_dynamic:
                    if math.sqrt((ob_x - ob_d.x) ** 2 + (ob_y - ob_d.y) ** 2) < 0.175:
                        flag = False
                        break
                if flag:
                    break
            self.obs_dynamic.append(obstacle_dynamic(self, ob_x, ob_y, 0.05)) #半径参数为0.05
            theta_temp += self.obs_dynamic[i].theta
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic #障碍物的平均角速度

        #目标位置
        self.target = []
        for i in range(self.n_uav):
            if flag_vertical:
                tar_x = (0.5 + i) / self.n_uav #这将使目标均匀分布
                # tar_x = 0.5
                if flag_up_right:
                    tar_y = random.uniform(0.95, 1)
                else:
                    tar_y = random.uniform(0, 0.05)
            else:
                tar_y = (0.5 + i) / self.n_uav
                if flag_up_right:
                    tar_x = random.uniform(0.95, 1)
                else:
                    tar_x = random.uniform(0, 0.05)
            self.target.append([tar_x, tar_y])

        self.uavs = []
        self.hover_flag = []
        if flag_vertical:
            uav_x = random.uniform(0.4, 0.6)
            if flag_up_right:
                uav_y = random.uniform(0.02, 0.08)
            else:
                uav_y = random.uniform(0.92, 0.98)
        else:
            uav_y = random.uniform(0.4, 0.6)
            if flag_up_right:
                uav_x = random.uniform(0.02, 0.08)
            else:
                uav_x = random.uniform(0.92, 0.98)
        for i in range(self.n_uav):
            if flag_vertical:
                self.uavs.append(UAV(self, i, uav_x + (i-self.n_uav//2) * 0.05, uav_y, self.target[i]))
            else:
                self.uavs.append(UAV(self, i, uav_x, uav_y + (i-self.n_uav//2) * 0.05, self.target[i]))
            self.hover_flag.append(False)

        self.tu_service_state=[]
        state = []
        for uav in self.uavs:
            state.append(uav.state()) #无人机的状态信息
            self.tu_service_state.append(0)
        return state

    def reset_episode(self, flag_test = False):
        flag_vertical = True#round(random.random())
        flag_up_right = True#round(random.random())

        # 移动障碍物
        n_ob_dy = self.n_ob_dynamic if flag_test else np.random.randint(3, 8)
        self.obs_dynamic = []
        theta_temp = 0 #累积移动障碍物的角速度
        for i in range(n_ob_dy):
            while (1):
                ob_x = random.uniform(0.15, 0.85)
                ob_y = random.uniform(0.15, 0.85)
                flag = True
                for ob_d in self.obs_dynamic:
                    if math.sqrt((ob_x - ob_d.x) ** 2 + (ob_y - ob_d.y) ** 2) < 0.175:
                        flag = False
                        break
                if flag:
                    break
            self.obs_dynamic.append(obstacle_dynamic(self, ob_x, ob_y, 0.05)) #半径参数为0.05
            theta_temp += self.obs_dynamic[i].theta
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic #障碍物的平均角速度

        #目标位置
        self.target = []
        for i in range(self.n_uav):
            if flag_vertical:
                # tar_x = (0.5 + i) / self.n_uav #这将使目标均匀分布
                tar_x = 0.5
                if flag_up_right:
                    tar_y = random.uniform(0.95, 1)
                else:
                    tar_y = random.uniform(0, 0.05)
            else:
                tar_y = (0.5 + i) / self.n_uav
                if flag_up_right:
                    tar_x = random.uniform(0.95, 1)
                else:
                    tar_x = random.uniform(0, 0.05)
            self.target.append([tar_x, tar_y])

        self.uavs = []
        self.hover_flag = []
        if flag_vertical:
            uav_x = random.uniform(0.4, 0.6)
            if flag_up_right:
                uav_y = random.uniform(0.02, 0.08)
            else:
                uav_y = random.uniform(0.92, 0.98)
        else:
            uav_y = random.uniform(0.4, 0.6)
            if flag_up_right:
                uav_x = random.uniform(0.02, 0.08)
            else:
                uav_x = random.uniform(0.92, 0.98)
        for i in range(self.n_uav):
            if flag_vertical:
                self.uavs.append(UAV(self, i, uav_x + (i-self.n_uav//2) * 0.05, uav_y, self.target[i]))
            else:
                self.uavs.append(UAV(self, i, uav_x, uav_y + (i-self.n_uav//2) * 0.05, self.target[i]))
            self.hover_flag.append(False)

        self.tu_service_state=[]
        state = []
        for uav in self.uavs:
            state.append(uav.state()) #无人机的状态信息
            self.tu_service_state.append(0)
        return state

    #可视化环境状态
    def render(self, flag = 0):
        if flag:
            plt.clf()
            plt.xlim(-0.05*self.length, 1.05*self.length)
            plt.ylim(-0.05*self.width, 1.05*self.width)
            #目标位置
            for uav in self.uavs:
                plt.scatter(uav.x_tar * self.length, uav.y_tar * self.width, s=20, color='r', marker='x')
            #固定障碍物
            for ob in self.obs:
                #plt.scatter(ob.x * self.length, ob.y * self.width, s=20, color='black', marker='o')
                theta = np.arange(0, 2*np.pi, 0.01)
                x = ob.x * self.length + ob.r * self.length * np.cos(theta)
                y = ob.y * self.width + ob.r * self.length * np.sin(theta)
                # plt.plot(x, y, color = 'black')
                plt.fill(x, y, ob.r, color='black')

                x = ob.x * self.length + (ob.r+0.03) * self.length * np.cos(theta)
                y = ob.y * self.width + (ob.r+0.03) * self.length * np.sin(theta)
                #plt.plot(x, y, 'k--')

            ''''#画威胁梯度图
            map_threat = np.ones((self.length, self.width))
            for i in range(self.length):
                for j in range(self.width):
                    for ob in self.obs:
                        d_2 = (i/self.length - ob.x) ** 2 + (j/self.width - ob.y) ** 2
                        map_threat[i][j] = map_threat[i][j] * (1 - math.exp(- d_2 / (2*ob.rr)))
                    for ob_d in self.obs_dynamic:
                        d_2 = (i/self.length - ob_d.x) ** 2 + (j/self.width - ob_d.y) ** 2
                        map_threat[i][j] = map_threat[i][j] * (1 - math.exp(- d_2 / (2*ob_d.rr)))
                    map_threat[i][j] = (1 - map_threat[i][j])/0.8
            X, Y = np.meshgrid(np.arange(0, self.length, 1), np.meshgrid(np.arange(0, self.width, 1)))
            Z = [[0 for _ in range(self.length)] for _ in range(self.width)]
            for i in range(self.length):
                for j in range(self.width):
                    Z[X[i][j]][Y[i][j]] = map_threat[i][j]
            plt.contour(X, Y, Z, [0.2, 0.5, 0.7, 0.9, 0.95])
            #countour = plt.contour(X, Y, Z)
            #plt.clabel(countour, fontsize=10, colors='r')'''
        #移动障碍物
        for ob_d in self.obs_dynamic:
            plt.scatter(ob_d.x * self.length, ob_d.y * self.width, s=10, color='black', marker='^')
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = ob_d.x * self.length + ob_d.r * self.length * np.cos(theta)
            y = ob_d.y * self.width + ob_d.r * self.length * np.sin(theta)
            #plt.plot(x, y, color='gray')
            plt.fill(x, y, ob_d.r, color='gray')
            x = ob_d.x * self.length + (ob_d.r + 0.03) * self.length * np.cos(theta)
            y = ob_d.y * self.width + (ob_d.r + 0.03) * self.length * np.sin(theta)
            #plt.plot(x, y, 'k--')
        #终端用户
        for tu in self.tus:
            #终端用户已完成任务，则用绿色标记表示，否则用棕色标记表示
            plt.scatter(tu.x * self.length, tu.y * self.width, s=10, color='green' if tu.flag_done else 'brown', marker='*')
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = tu.x * self.length + tu.r * self.length * np.cos(theta)
            y = tu.y * self.width + tu.r * self.length * np.sin(theta)
            plt.plot(x, y, color='green' if tu.flag_done else 'brown')
            plt.text((tu.x+0.01)*self.length, (tu.y+0.01)*self.width, '%.2f'%tu.I)

        #无人机
        for uav in self.uavs:
            plt.scatter(uav.x * self.length, uav.y * self.width, s=10, color='b', marker='^')
        plt.pause(0.01)




    #用于推进仿真环境的一个时间步。它接受一个动作列表，用于更新无人机的状态。
    def step(self, action):
        theta_temp = 0
        for ob_d in self.obs_dynamic:
            ob_d.update()
            theta_temp += ob_d.theta
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic
    #计算无人机的奖励和下一个状态
        reward = []
        next_state = []
        # done = True
        done = []
        # hover_flag = []
        collected_data = []
        for i in range(self.n_uav):
            done.append(True)
            # hover_flag.append(False)
            collected_data.append(0)
        done = np.array(done, dtype=bool)

        #处理终端用户和无人机之间的通信
        for uav in self.uavs:
            if uav.done:
                continue
            for tu in self.tus:
                if tu.flag_done:
                    continue
                d = math.sqrt((uav.x - tu.x)**2 + (uav.y - tu.y)**2)
                # if d < tu.r:
                #     hover_flag[uav.id] = True
                #     tu.I -= 1 #数据量-1
                #     if tu.I <= 0:
                #         tu.flag_done = True
                #         hover_flag[uav.id] = False
                if d < tu.r:
                    self.hover_flag[uav.id] = True
                    dir = math.sqrt((uav.x - tu.x) ** 2 + (uav.x - uav.y) ** 2 + uav.H ** 2)

                    theta_tu = np.arcsin(uav.H / dir)
                    theta_tu = fix_theta(theta_tu , -1,1)
                    uav.pro = 1 / (1 + uav.alpa * math.exp(-uav.beta * (theta_tu- uav.alpa)))
                    # 路径损耗
                    uav.h = uav.K0 * dir ** 2 * (uav.pro * uav.mu_los + (1 - uav.pro) * uav.mu_nlos)
                    # 数据采集速率
                    S = uav.P / (uav.noise * uav.h)
                    rate_tu = uav.B * math.log(1 + S, 2)  # 香农公式
                    collected_data[uav.id] += tu.I
                    R_max = uav.B * math.log((1 + uav.P / (uav.noise * uav.K0 * uav.H * uav.H * uav.mu_los)), 2)
                    rate_tu = rate_tu / R_max
                    if tu.I > 0:
                        tu.I -= rate_tu
                    if tu.I <= 0:
                        tu.I = 0
                        tu.flag_done = True
                        self.tu_service_state[uav.id]+=1
                        self.hover_flag[uav.id] = False


        for i in range(self.n_uav):
            if self.uavs[i].done:
                reward.append(0)
                next_state.append([])
                continue
            else:

                reward_t, done_t = self.uavs[i].update(action[i] , self.hover_flag[i], collected_data[i])
                reward.append(reward_t)
                if done_t == False:
                    done[i] = False
                next_state.append(self.uavs[i].state())


        return next_state, reward, done


if __name__ == '__main__':
    env = ENV()

    '''for _ in range(10000):
        env.reset()
        env.render(1)
        #plt.pause(1)'''
    env.reset(flag_test=True)
    # for _ in range(100):
    #     env.reset(flag_test=True)
    #     # env.render(1)
    #     # uav_x = []
    #     # uav_y = []
    #     # ob_x = []
    #     # ob_y = []
    #     # for _ in range(env.n_uav):
    #     #     uav_x.append([])
    #     #     uav_y.append([])
    #     # for _ in range(env.n_ob_dynamic):
    #     #     ob_x.append([])
    #     #     ob_y.append([])
    #     # while 1:
    #     #     action = []
    #     #     for uav in env.uavs:
    #     #         action.append(5)
    #     #         state = uav.state()
    #     #     nt, r, done = env.step(action) #将动作列表 action 传递给仿真环境，然后获取下一个状态 nt、奖励 r 和完成标志 done。
    #     #     env.render(1)
    #     #     # print(nt)
    #     #     for i in range(env.n_uav):
    #     #         if action[i] != -1:
    #     #             uav_x[i].append(env.uavs[i].x * env.length)
    #     #             uav_y[i].append(env.uavs[i].y * env.width)
    #     #     for ob_d in env.obs_dynamic:
    #     #         ob_x[i].append(ob_d.x * env.length)
    #     #         ob_y[i].append(ob_d.y * env.width)
    #     #     if done:
    #     #         break
    #     #     plt.pause(0.1)
    #     # for i in range(env.n_uav):
    #     #     if env.uavs[i].d_tar < (2 * env.uavs[i].stepWay):
    #     #         uav_x[i].append(env.uavs[i].x_tar * env.length)
    #     #         uav_y[i].append(env.uavs[i].y_tar * env.width)
    #     #     plt.plot(uav_x[i], uav_y[i])
    #     # # for i in range(env.n_ob_dynamic):
    #     # #     plt.plot(ob_x[i], ob_y[i], 'r-.')
    #     # plt.pause(10)
