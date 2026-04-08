import numpy as np
import random
import math
import matplotlib.pyplot as plt
from UAV import UAV
#from UAV import UAV_biginput
from TerminalUser import TerminalUser
from obstacle import obstacle
from obstacle import obstacle_dynamic
from mpl_toolkits.mplot3d import Axes3D

def fix_theta(theta, down, up):
    t = theta
    d = up - down
    t += d if t < down else 0
    t -= d if t > up else 0
    return t

def draw_cylinder(x, y, h, r, resolution, ax):
    # Create a meshgrid for the cylinder
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, h, resolution)
    theta, Z = np.meshgrid(theta, z)

    # Parametric equations for the cylinder
    X = r * np.cos(theta) + x
    Y = r * np.sin(theta) + y

    # Plot the cylinder
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color='k', alpha=0.4)

    z_top = h
    z_bottom = 0
    theta_disc = np.linspace(0, 2 * np.pi, resolution)
    X_disc = r * np.cos(theta_disc) + x
    Y_disc = r * np.sin(theta_disc) + y

    # Plot top and bottom circles
    ax.plot_trisurf(X_disc, Y_disc, [z_top] * resolution, color='k', alpha=0.75)  # Top surface
    ax.plot_trisurf(X_disc, Y_disc, [z_bottom] * resolution, color='k', alpha=0.4)  # Bottom surface

class ENV():
    def __init__(self):
        self.length = 1000
        self.width = 1000
        self.x_bound = 1
        self.y_bound = 1
        self.n_uav = 3 #目标位置的数量
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
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.collaboration = True

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
                ob_x = random.uniform(0.15, 0.85*self.x_bound) #障碍物的位置必须在 (0.15, 0.85) 的范围内
                ob_y = random.uniform(0.15, 0.85*self.y_bound)
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

        self.tu_x = []
        self.tu_y = []
        #终端用户
        n_tus = self.n_tus if flag_test else np.random.randint(9, 36)
        self.tus = []
        for i in range(n_tus):#在随机的位置 (tu_x, tu_y) 创建终端用户
            while 1:
                if flag_vertical:
                    tu_x = random.uniform(0, 1*self.x_bound)
                    tu_y = random.uniform(0.2, 0.8*self.y_bound)
                else:
                    tu_x = random.uniform(0.2, 0.8*self.x_bound)
                    tu_y = random.uniform(0, 1*self.y_bound)
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
            self.tu_x.append(self.length - tu_x * self.length)
            self.tu_y.append(tu_y * self.width)

        self.d_ob_x = []
        self.d_ob_y = []
        self.d_ob_z = []
        self.d_ob_x_init = []
        self.d_ob_y_init = []
        self.d_ob_z_init = []
        # 移动障碍物
        n_ob_dy = self.n_ob_dynamic if flag_test else np.random.randint(2, 9)
        self.obs_dynamic = []
        theta_temp = 0 #累积移动障碍物的角速度
        for i in range(n_ob_dy):
            while (1):
                ob_x = random.uniform(0.15, 0.85*self.x_bound)
                ob_y = random.uniform(0.15, 0.85*self.y_bound)
                flag = True
                for ob_d in self.obs_dynamic:
                    if math.sqrt((ob_x - ob_d.x) ** 2 + (ob_y - ob_d.y) ** 2) < 0.175:
                        flag = False
                        break
                if flag:
                    break
            self.obs_dynamic.append(obstacle_dynamic(self, ob_x, ob_y, 0.05)) #半径参数为0.05
            theta_temp += self.obs_dynamic[i].theta
            self.d_ob_x.append([])
            self.d_ob_y.append([])
            self.d_ob_z.append([])
            self.d_ob_x_init.append((1 - ob_x) * self.length)
            self.d_ob_y_init.append(ob_y * self.length)
            self.d_ob_z_init.append(100)
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic #障碍物的平均角速度

        #目标位置
        self.target = []
        self.target_x = []
        self.target_y = []
        for i in range(self.n_uav):
            if flag_vertical:
                # tar_x = (0.5 + i) / self.n_uav #这将使目标均匀分布
                tar_x = 0.5*self.x_bound
                if flag_up_right:
                    tar_y = random.uniform(0.95*self.y_bound, 1*self.y_bound)
                else:
                    tar_y = random.uniform(0*self.y_bound, 0.05*self.y_bound)
            else:
                tar_y = (0.5 + i) / self.n_uav *self.y_bound
                if flag_up_right:
                    tar_x = random.uniform(0.95*self.y_bound, 1*self.x_bound)
                else:
                    tar_x = random.uniform(0*self.y_bound, 0.05*self.x_bound)
            self.target.append([tar_x, tar_y])
            self.target_x.append(tar_x * self.length)
            self.target_y.append(tar_y * self.width)

        self.uavs = []
        if flag_vertical:
            uav_x = random.uniform(0.4*self.x_bound, 0.6*self.x_bound)
            if flag_up_right:
                uav_y = random.uniform(0, 0.02)
            else:
                uav_y = random.uniform(0.92*self.y_bound, 0.98*self.y_bound)
        else:
            uav_y = random.uniform(0.4*self.y_bound, 0.6*self.y_bound)
            if flag_up_right:
                uav_x = random.uniform(0, 0.02)
            else:
                uav_x = random.uniform(0.92*self.x_bound, 0.98*self.x_bound)
        for i in range(self.n_uav):
            if flag_vertical:
                self.uavs.append(UAV(self, i, uav_x + (i-self.n_uav//2) * 0.05, uav_y, self.target[i]))
            else:
                self.uavs.append(UAV(self, i, uav_x, uav_y + (i-self.n_uav//2) * 0.05, self.target[i]))

        self.tu_service_state=[]
        uav_state = []
        near_uav_id = []
        for uav in self.uavs:
            state, near_id = uav.state()
            uav_state.append(state) #无人机的状态信息
            near_uav_id.append(near_id)
            self.tu_service_state.append(0)
        return uav_state, near_uav_id

    #可视化环境状态
    def render(self, flag = 0):
        if flag:
            plt.clf()
            plt.xlim(-0.05*self.length*self.x_bound, 1.05*self.length*self.x_bound)
            plt.ylim(-0.05*self.width*self.y_bound, 1.05*self.width*self.y_bound)
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

    def render_3D(self, uav_x, uav_y, uav_z):
        # 地理坐标  //  栅格坐标的范围为3713*3688
        # x = np.linspace(start=0, stop=self.length, num=int(self.length / 1))  # 开始，终止，点数
        # y = np.linspace(start=0, stop=self.width, num=int(self.length / 1))
        # Z = np.zeros((int(self.length / 1), int(self.length / 1)))
        # np.meshgrid函数输入横纵坐标向量，输出横纵 坐标矩阵（用来描述网格点）
        # X, Y = np.meshgrid(x, y)  # x,y都是多维矩阵，对应位置的值组成一个坐标，可以使用对应下标访问
        # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))\
        fig = plt.figure(figsize=(10,9))
        ax = Axes3D(fig)
        fig.add_axes(ax)
        # for ob in self.obs:
        #     for i, x_i in enumerate(x):
        #         for j, y_j in enumerate(y):
        #             if math.sqrt((x_i / self.length - ob.x) ** 2 + (y_j / self.width - ob.y) ** 2) <= ob.r:
        #                 Z[(self.length-i), j] = 0.2 * self.length

        # 自定义颜色映射，确保渐变更加明显
        # colors = [(1.0, 1.0, 1.0), (0.8, 0.8, 0.8), (0.6, 0.6, 0.6), (0.2, 0.2, 0.2)]  # 深灰色到浅灰色再到白色
        # n_bins = 256  # 使用更多的颜色层级
        # cmap_name = 'snow_mountain'
        # custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        # ls = LightSource(270, 90)
        # rgb = ls.shade(Z, cmap=custom_cmap, vert_exag=0.1, blend_mode='soft')

        for i, ob in enumerate(self.obs):
            draw_cylinder(ob.y*self.length, (1-ob.x)*self.length, 200, ob.r*self.length, 100, ax)

        ax.scatter3D(self.tu_y, self.tu_x, np.ones(self.n_tus)*0, color='green', marker='^', s=80, alpha=1,zorder=10)
        # ax.scatter3D(self.target_y, self.target_x, np.ones(self.n_uav) * 5, color='red', marker='o', s=30, alpha=1, linewidths=5, zorder=10)
        ax.scatter3D(self.d_ob_y_init, self.d_ob_x_init, self.d_ob_z_init, color='red', marker='o', s=30, alpha=1,
                     linewidths=5, zorder=10)
        for i in range(self.n_ob_dynamic):
            ax.plot3D(self.d_ob_y[i], self.d_ob_x[i], self.d_ob_z[i], 'r-.', zorder=10)
        ax.plot3D(uav_x[0], uav_y[0], uav_z[0], c='blue', label='UAV_0', zorder=10)
        ax.plot3D(uav_x[1], uav_y[1], uav_z[1], c='blue', label='UAV_1', zorder=10)
        ax.plot3D(uav_x[2], uav_y[2], uav_z[2], c='blue', label='UAV_2', zorder=10)
        # surf = ax.plot_surface(X, Y, Z, alpha=0.4, rstride=1, cstride=1, cmap=custom_cmap, linewidth=0, antialiased=False,
        #                        shade=False, zorder=1)
        ax.view_init(elev=65, azim=180)
        # plt.axis('off')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

    #用于推进仿真环境的一个时间步。它接受一个动作列表，用于更新无人机的状态。
    def step(self, action):
        theta_temp = 0
        for i, ob_d in enumerate(self.obs_dynamic):
            ob_d.update()
            self.d_ob_x[i].append((1 - ob_d.x) * self.length)
            self.d_ob_y[i].append(ob_d.y * self.length)
            self.d_ob_z[i].append(100)
            theta_temp += ob_d.theta
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic
    #计算无人机的奖励和下一个状态
        reward = []
        next_state = []
        near_uav_id = []
        # done = True
        done = []
        hover_flag = []
        for i in range(self.n_uav):
            done.append(True)
            hover_flag.append(False)
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
                    hover_flag[uav.id] = True
                    dir = math.sqrt((uav.x - tu.x) ** 2 + (uav.x - uav.y) ** 2 + uav.H ** 2)

                    theta_tu = np.arcsin(uav.H / dir)
                    theta_tu = fix_theta(theta_tu , -1,1)
                    uav.pro = 1 / (1 + uav.alpa * math.exp(-uav.beta * (theta_tu- uav.alpa)))
                    # 路径损耗
                    uav.h = uav.K0 * dir ** 2 * (uav.pro * uav.mu_los + (1 - uav.pro) * uav.mu_nlos)
                    # 数据采集速率
                    S = uav.P / (uav.noise * uav.h)
                    rate_tu = uav.B * math.log(1 + S, 2)  # 香农公式
                    R_max = uav.B * math.log((1 + uav.P / (uav.noise * uav.K0 * uav.H * uav.H * uav.mu_los)), 2)
                    rate_tu = rate_tu / R_max
                    if tu.I > 0:
                        tu.I -= rate_tu
                    if tu.I <= 0:
                        tu.I = 0
                        tu.flag_done = True
                        uav.iot_service_cnt += 1
                        self.tu_service_state[uav.id]+=1
                        hover_flag[uav.id] = False
        for i in range(self.n_uav):
            if self.uavs[i].done:
                reward.append(0)
                next_state.append([])
                near_uav_id.append([])
                continue
            else:
                reward_t, done_t = self.uavs[i].update(action[i] , hover_flag[i])
                reward.append(reward_t)
                if done_t == False:
                    done[i] = False
                state, near_id = self.uavs[i].state()
                next_state.append(state)
                near_uav_id.append(near_id)
        if self.collaboration:
            iot_state = []
            for tu in self.tus:
                if tu.flag_done:
                    continue
                covered = False
                for id_s in range(self.n_uav):
                    if self.uavs[id_s].done:
                        continue
                    d = math.sqrt((self.uavs[id_s].x - tu.x) ** 2 + (self.uavs[id_s].y - tu.y) ** 2)
                    if d < self.uavs[id_s].senser_tu_r:
                        covered = True
                if covered:
                    iot_state.append(1)
                else:
                    iot_state.append(0)
            numerator = 0
            denominator = 0
            for i, val in enumerate(iot_state):
                numerator += val
                denominator += val ** 2
            if denominator != 0:
                fairness_iot = numerator ** 2 / denominator / len(iot_state)
            else:
                fairness_iot = 0

            cnt = 0
            for i, uav in enumerate(self.uavs):
                numerator = 0
                denominator = 0
                if uav.done:
                    if uav.iot_service_rwd >= 0:
                        numerator += uav.iot_service_rwd
                        denominator += uav.iot_service_rwd ** 2
                        cnt += 1
            # cnt = 1
            # for id_s in near_uav_id[uav.id]:
            #     if self.uavs[id_s].iot_service_rwd >= 0 and id_s != -1:
            #         numerator += self.uavs[id_s].iot_service_rwd
            #         denominator += self.uavs[id_s].iot_service_rwd ** 2
            #         cnt += 1
            if denominator != 0 and cnt != 0:
                fairness_uav = numerator ** 2 / denominator / cnt
            else:
                fairness_uav = 0

            for i, uav in enumerate(self.uavs):
                if uav.done:
                    reward[uav.id] += fairness_iot + fairness_uav

        return next_state, reward, done, near_uav_id


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
