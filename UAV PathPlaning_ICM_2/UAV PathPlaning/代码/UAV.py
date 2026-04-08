import math
import numpy as np
from TerminalUser import TerminalUser


def fix_theta(theta, down, up):
    t = theta
    d = up - down
    t += d if t < down else 0
    t -= d if t > up else 0
    return t

def f1(x):     #TU奖励权重函数， 距离x越小，权重越大, x:[0,1], y:[0.5, 1]
    y = np.exp(- (x * 1.7) ** 2) * 0.5 + 0.5
    return y

def f2(x):     #TU需上传数据量权重函数， 数据量越大，权重越大， x:[2, 10]为整数， y:[0.5, 1]
    y = 1 - np.exp(- x ** 3 / (x + 75))
    y = y * 0.5 + 0.5
    return y

def distance_2d(x,y):
    dist=math.sqrt(x**2+y**2)
    return dist

class UAV():
    def __init__(self, env, id, x, y, tar):
        self.strict_flag = 0
        self.id = id
        self.env = env
        self.num_actions = env.num_actions
        self.num_lidar = env.num_lidar
        self.maxr = env.maxr
        self.x = x
        self.y = y
        self.H = 0.1 #无人机飞行高度
        self.done = False
        self.stepWay = 0.01
        self.senser_ob_r = 0.3
        self.senser_tu_r = 0.5
        self.p_move = 1
        self.step_done = 0
        self.energy = int(4 / self.stepWay)
        self.x_tar = tar[0]
        self.y_tar = tar[1]
        self.d_tar = math.sqrt((self.x - self.x_tar) ** 2 + (self.y - self.y_tar) ** 2) #距离目标点的距离
        #
        self.dir_tar = math.atan2(self.y_tar - self.y, self.x_tar - self.x) #与目标点的方向角
        self.theta = self.dir_tar
        # 可能的服务目标
        self.num_tu_can = env.num_tu_can
        self.N_POI = env.num_tu_can
        self.pro = [] #视距概率
        self.alpa = 10  #LOS概率控制量
        self.beta = 0.6
        self.h = [] #路径损耗
        self.fc = 1.5e8 #载波频率
        self.c = 3e8 #光速
        self.K0 = 16 * (math.pi ** 2) * (self.fc ** 2) /(self.c ** 2) #路径损耗系数K0
        self.mu_los = 1 #los链路衰减因子
        self.mu_nlos = 1 #nlos链路衰减因子
        self.noise = pow(10,-12) #噪声功率为-90dbm
        self.B = 1 #带宽
        self.P = pow(10, -5)  # 传感器的发射功率 0.01mW,-20dbm

        self.tu_can = []
        self.d_tu = []
        self.data_rate = [] #数据传输速率
        self.id_nearlist = []
        self.get_tu_can()
        #无人机之间通信的路劲损耗阈值
        self.h_free_up = 1

        self.lidar = np.ones(self.num_lidar)
        self.get_lidar()

        self.collision = False
    #获取无人机周围的终端用户LOT节点，并存储在tu_can列表中
    def get_tu_can(self):
        self.tu_can = []  # 可能的服务目标
        self.d_tu = []
        self.data_rate = []

        for _ in range(self.num_tu_can):
            self.tu_can.append(TerminalUser(-1, self.x_tar, self.y_tar, 0))
            self.d_tu.append(self.senser_tu_r)
            self.data_rate.append(0)
        #遍历环境中的终端用户，计算无人机与终端用户的距离，如果距离在传感器范围内，且没有其他无人机已经服务该用户终端，则将终端用户添加到tu_can列表中
        for tu in self.env.tus:
            if tu.flag_done:
                continue
            d = math.sqrt((self.x - tu.x) ** 2 + (self.y - tu.y) ** 2 + self.H ** 2) #无人机与服务节点的距离

            # 仰角
            theta_tu_high = np.arcsin(self.H / d)
            theta_tu_high = fix_theta(theta_tu_high, -1, 1)  # 归一化
            # 视距概率
            self.pro = 1 / (1 + self.alpa * math.exp(-self.beta * (theta_tu_high - self.alpa)))
            # 路径损耗
            self.h = self.K0 * d ** 2 * (self.pro * self.mu_los + (1 - self.pro) * self.mu_nlos)
            # 数据采集速率
            S = self.P / (self.noise * self.h)
            rate_tu = self.B * math.log(1 + S, 2)  # 香农公式

            if d <= self.senser_tu_r:
                wast = d + math.sqrt((self.x_tar - tu.x) ** 2 + (self.y_tar - tu.y) ** 2)    #无人机与服务节点之间的距离+目标点与服务节点的距离
                flag = True

                #考虑了其他无人机对终端用户的争抢情况，并选择了一个代价最小的终端用户来服务
                for uav in self.env.uavs:
                    if uav.id != self.id and uav.done == False and math.sqrt((uav.x - tu.x) ** 2 + (uav.y - tu.y) ** 2) < uav.senser_tu_r:
                        wast_t = math.sqrt((uav.x - tu.x) ** 2 + (uav.y - tu.y) ** 2) + \
                                 math.sqrt((uav.x_tar - tu.x) ** 2 + (uav.y_tar - tu.y) ** 2)
                        if wast_t < wast:
                            flag = False
                            break
                if flag:
                    self.tu_can.append(tu)
                    self.data_rate.append(rate_tu)
                    self.d_tu.append(d)

        # self.allocation()

        self.id_nearlist = sorted(range(len(self.d_tu)), key=lambda k: self.d_tu[k])

    def allocation(self):
        tu_can_other = []
        d_tu_other = []
        uav_other = []
        allocation_flag = False
        for uav in self.env.uavs:
            if uav.done == False and uav.id!=self.id:
                d = math.sqrt((self.x - uav.x) ** 2 + (self.y - uav.y) ** 2)
                if d <= self.senser_tu_r:
                    allocation_flag = True
                    tu_can_other.append(uav.tu_can)
                    d_tu_other.append(uav.d_tu)
                    uav_other.append(uav)
        if allocation_flag == True:
            tu_remove_id = []
            for tu_self_id, tu_self in enumerate(self.tu_can):
                if tu_self.id != -1:
                    d_self = self.d_tu[tu_self_id]
                    for uav_id, uav in enumerate(uav_other):
                        for tu_id, tu in enumerate(tu_can_other[uav_id]):
                            if tu.id == tu_self.id:
                                if d_self >= d_tu_other[uav_id][tu_id]:
                                    # self.d_tu[tu_self_id]+=self.senser_tu_r
                                    tu_remove_id.append(tu_self_id)
            tu_can_rmv = []
            d_tu_rmv = []
            datarate_rmv = []
            for rmv in tu_remove_id:
                tu_can_rmv.append(self.tu_can[rmv])
                d_tu_rmv.append(self.d_tu[rmv])
                datarate_rmv.append(self.data_rate[rmv])
            self.tu_can = list(filter(lambda x: x not in tu_can_rmv, self.tu_can))
            self.d_tu = list(filter(lambda x: x not in d_tu_rmv, self.d_tu))
            self.data_rate = list(filter(lambda x: x not in datarate_rmv, self.data_rate))

    #激光雷达扫描，获取周围障碍物信息
    def get_lidar(self):
        self.lidar = np.ones(self.num_lidar) #存储雷达测量结果，初始化测量值为1
        for i in range(self.num_lidar):
            theta = self.theta - math.pi + i * 2 * math.pi / self.num_lidar  #当前迭代角度 i 对应的极坐标角度 theta，以便确定激光雷达的测量方向
            l = 0 #在每个角度迭代开始时，将距离 l 初始化为0，用于记录激光测量的距离
            for _ in range(int(self.senser_ob_r / self.stepWay) + 1): #在当前角度下进行测量，不断延长测量距离l，判断测量点是否位于障碍物范围
                l_x = self.x + l * math.cos(theta) #激光雷达测量点的坐标，self.x为无人机当前的位置
                l_y = self.y + l * math.sin(theta)
                if l_x < -0.1 or l_x > 1.1 or l_y < -0.1 or l_y > 1.1: #超出地图边界，测量无效
                    l = self.senser_ob_r
                    break
                flag = False
                #无人机与静态障碍物发生碰撞，即距离d小于障碍物半径，则将flag设置为True
                for ob in self.env.obs:
                    if abs(ob.x - l_x) > ob.r or abs(ob.y - l_y) > ob.r:
                        continue
                    d = math.sqrt((ob.x - l_x) ** 2 + (ob.y - l_y) ** 2)
                    if d < ob.r:
                        flag = True
                        break
                #如果flag设置为True，则表示当前角度下存在障碍物，退出循环
                if flag:
                    break
                #无人机与动态障碍物发生碰撞
                for ob_d in self.env.obs_dynamic:
                    if abs(ob_d.x - l_x) > ob_d.r or abs(ob_d.y - l_y) > ob_d.r:
                        continue
                    d = math.sqrt((ob_d.x - l_x) ** 2 + (ob_d.y - l_y) ** 2)
                    if d < ob_d.r:
                        flag = True
                        break
                if flag:
                    break
                #l 表示当前测量点与无人机位置的距离，通过每次将 self.stepWay（步进距离）累加到 l 上
                l += self.stepWay
            # 测量结果被归一化
            self.lidar[i] = l / self.senser_ob_r


    #该方法用于获取当前无人机的状态信息，包括无人机与目标点的方向和距离，以及周围终端用户、其他无人机和障碍物的信息。
    def state(self):
        dir_tar = (self.theta - self.dir_tar) / math.pi  #无人机当前的朝向角度
        dir_tar = fix_theta(dir_tar, -1, 1)  #归一化到-1到1之间
        state_grid = [dir_tar, self.d_tar] # self.d_tar距离目标点的距离

        #探测范围内最近的三个TU的方向、距离， 不足三个补目标点方向
        #最近且代价最小的tu
        for i in range(self.num_tu_can):
            #无人机位置和特定终端用户位置之间的方向角
            dir_tu_before = math.atan2(self.tu_can[self.id_nearlist[i]].y - self.y, self.tu_can[self.id_nearlist[i]].x - self.x)
            #方向角归一化
            dir_tu = (self.theta - dir_tu_before) / math.pi
            dir_tu = fix_theta(dir_tu, -1, 1)

        # #无人机与用户之间的通信模型
        #
        #     #计算无人机与终端用户之间的距离
        #     dis_to_tu = math.sqrt((self.tu_can[self.id_nearlist[i]].x - self.x)**2 + (self.tu_can[self.id_nearlist[i]].y - self.y)**2)
        #     #仰角
        #     theta_tu_high = 180 / math.pi * np.arcsin(self.H / dis_to_tu)
        #     theta_tu_high = fix_theta(theta_tu_high, -1, 1)
        #     #视距概率
        #     self.pro = 1/(1 + self.alpa * math.exp(-self.beta*(theta_tu_high - self.alpa)))
        #     #路径损耗
        #     self.h = self.K0 * dis_to_tu**2 * (self.pro * self.mu_los + (1-self.pro) * self.mu_nlos)
        #     #数据采集速率
        #     S = self.P / (self.noise * self.h)
        #     rate_tu = self.B * math.log(1 + S , 2) #香农公式

            state_grid.append(dir_tu)
            state_grid.append(self.d_tu[self.id_nearlist[i]] / self.senser_tu_r)
            state_grid.append(self.tu_can[self.id_nearlist[i]].I / self.env.maxI)
            # state_grid.append(rate_tu)



        #探测范围内最近的2架其他无人机方向、距离，不足两架补方向：与运动方向相反、距离为1

        '''for uav in self.env.uavs:
            if uav.id != self.id:
                d = math.sqrt((self.x - uav.x)**2 + (self.y - uav.y)**2)
                if d < self.senser_ob_r:
                    dir = math.atan2(uav.y - self.y, uav.x - self.x)
                    dir = fix_theta((self.theta - dir)/math.pi, -1, 1)
                    state_grid.append(dir)
                    state_grid.append(d/self.senser_ob_r)
                else:
                    state_grid.append(-1)
                    state_grid.append(1)'''
        d_uav = [2, 2] #存储当前无人机与其他无人机的距离
        id_uav = [-1, -1] #存储无人机的id
        n_uav = 2
        for uav in self.env.uavs:
            if uav.id != self.id:
                if abs(self.x - uav.x) > self.senser_ob_r or abs(self.y - uav.y) > self.senser_ob_r:
                    continue
                d = math.sqrt((uav.x - self.x) ** 2 + (uav.y - self.y) ** 2)
                if d < self.senser_ob_r:
                    d_uav.append(d)
                    id_uav.append(uav.id)
                    n_uav += 1
        i_uav = sorted(range(len(d_uav)), key=lambda k: d_uav[k]) #对感知范围内其他无人机的距离进行排序，以获取最近的两架无人机的索引
        for i in range(2):
            if id_uav[i_uav[i]] == -1: #如果最近的无人机没有ID（即初始化时的占位值），则表示没有其他无人机在感知范围内。
                state_grid.append(-1) #在这种情况下，将-1（表示没有无人机）和1（表示一个固定的距离）分别添加到 state_grid 中。
                state_grid.append(1)
            else:
                dir = math.atan2(self.env.uavs[id_uav[i_uav[i]]].y-self.y, self.env.uavs[id_uav[i_uav[i]]].x-self.x)
                dir = fix_theta((self.theta-dir)/math.pi, -1, 1)#计算当前无人机与最近无人机之间的方向，
                state_grid.append(dir) #添加方向信息
                state_grid.append(d_uav[i_uav[i]]/self.senser_ob_r) #添加距离信息

        #在一定范围内探测障碍物， 每个方向障碍物距离
        for i in range(self.num_lidar):
            state_grid.append(self.lidar[i])
            #state_grid.append(self.lidar[1][i])
        ''''# 最近的3个移动障碍物 方向，距离，运动方向
        d_ob_d = []
        for ob_d in self.env.obs_dynamic:
            d_ob_d.append(math.sqrt((self.x - ob_d.x) ** 2 + (self.y - ob_d.y) ** 2))
        id_ob_d = sorted(range(len(d_ob_d)), key=lambda k: d_ob_d[k])
        for i in range(3):
            if len(d_ob_d)>0 and d_ob_d[id_ob_d[i]] < self.senser_ob_r + self.env.obs_dynamic[0].r:
                dir = math.atan2(self.env.obs_dynamic[id_ob_d[i]].y - self.y,
                                 self.env.obs_dynamic[id_ob_d[i]].x - self.x)
                dir = fix_theta((self.theta - dir) / math.pi, -1, 1)
                state_grid.append(dir)
                state_grid.append((d_ob_d[id_ob_d[i]] - self.env.obs_dynamic[0].r) / self.senser_ob_r)
                dir_ob = (self.theta - self.env.obs_dynamic[id_ob_d[i]].theta) / math.pi
                dir_ob = fix_theta(dir_ob, -1, 1)
                state_grid.append(dir_ob)
            else:
                state_grid.append(-1)
                state_grid.append(1)
                state_grid.append(-1)'''

        return state_grid

    #根据传入的动作更新无人机的状态
    def update(self, action, hover):

        self.theta += action * math.pi / 2 / (self.num_actions - 1) - math.pi / 4 #更新角度
        self.theta = fix_theta(self.theta, -math.pi, math.pi)
        x_pre = self.x
        y_pre = self.y

        if hover:
            self.x = x_pre
            self.y = y_pre
        else:
            self.x = self.x + math.cos(self.theta) * self.stepWay  #更新坐标，沿着当前朝向前进
            self.y = self.y + math.sin(self.theta) * self.stepWay

        # state_update = self.state()
        # data_rate = state_update[5] #数据传输速率
        self.get_lidar() #获取激光雷达的扫描信息

        self.step_done += 1
        reward = 0

        self.energy -= self.p_move  #能耗
        reward -= 1 #每走一步，奖励值-1

        if self.energy <= 0:  #能量耗尽 ，表示无人机完成任务，无法再执行动作
            self.done = True

        #靠近目标点获得奖励， [-1, 1]
        d_tar_next = math.sqrt((self.x - self.x_tar) ** 2 + (self.y - self.y_tar) ** 2)
        flag_tu = True
        for tu in self.tu_can:
            if tu.id > 0:
                flag_tu &= tu.flag_done #检查是否所有的TU都已完成

        reward_tar = (self.d_tar - d_tar_next) / self.stepWay if flag_tu else 0.15*(self.d_tar - d_tar_next) / self.stepWay

        #靠近待服务的tu获得的奖励   加权和
        reward_tu = 0
        for i in range(self.num_tu_can):
            if self.id_nearlist[i] < self.num_tu_can:
                break
            d_tu = self.d_tu[self.id_nearlist[i]] #TU到无人机的距离
            date_rate = self.data_rate[self.id_nearlist[i]] #数据传输速率
            #下一时刻无人机与节点距离（已经update UAV坐标）
            d_tu_next = math.sqrt((self.x-self.tu_can[self.id_nearlist[i]].x)**2 + (self.y-self.tu_can[self.id_nearlist[i]].y)**2)
            '''if d_tu_next < self.tu_can[self.id_nearlist[i]].r:
                reward += 1'''


            weight = f1(d_tu) * f2(self.tu_can[self.id_nearlist[i]].I)  #f2表示数据量的权重，属性I为数据量
            reward_tu += weight * (d_tu - d_tu_next) / self.stepWay
            omga = 0.1
            R_max = self.B * math.log((1 + self.P / (self.noise * self.K0 * self.H * self.H * self.mu_los)) , 2)
            reward_tu += omga * date_rate / R_max

            #待服务节点的方向与目标点方向的夹角应不大于 90°
            dir_tu = math.atan2(self.tu_can[self.id_nearlist[i]].y-self.y, self.tu_can[self.id_nearlist[i]].x-self.x)
            if fix_theta(abs(self.dir_tar - dir_tu), 0, math.pi) > math.pi/2:
                reward_tu -= 0.5

        if len(self.tu_can) > self.num_tu_can:
            reward_tu = reward_tu / (len(self.tu_can) - self.num_tu_can)
        reward_tu = 1.15 * reward_tu

        #障碍物威胁
        reward_ob = 0
        delt = - (3*self.stepWay)**2 / math.log(1/3, math.e)
        for ob in self.env.obs:
            if abs(self.x - ob.x) > self.senser_ob_r + ob.r or abs(self.y - ob.y) > self.senser_ob_r + ob.r:
                continue
            d_ob = math.sqrt((self.x - ob.x) ** 2 + (self.y - ob.y) ** 2)
            if d_ob > self.senser_ob_r + ob.r:
                continue
            if d_ob <= ob.r:
                reward_ob -= 5

                if self.strict_flag:
                    self.done = True
                    self.collision = True
            else:
                if d_ob <= ob.r + 2 * self.stepWay: #无人机与动态障碍物之间的距离不是太近，但在感知范围内，
                    reward_ob -= 2
        for ob_d in self.env.obs_dynamic:
            if abs(self.x - ob_d.x) > self.senser_ob_r + ob_d.r or abs(self.y - ob_d.y) > self.senser_ob_r + ob_d.r:
                continue
            d_ob = math.sqrt((self.x - ob_d.x) ** 2 + (self.y - ob_d.y) ** 2)
            if d_ob > self.senser_ob_r + ob_d.r:
                continue
            if d_ob <= ob_d.r:
                reward_ob -= 5
                if self.strict_flag:
                    self.done = True
                    self.collision = True
            else:
                if d_ob <= ob_d.r+2*self.stepWay:
                    reward_ob -= 2

        #无人机之间的距离不能小于0.05，+ 连通性
        for uav in self.env.uavs:
            if uav.id != self.id and uav.done == False:
                d_uav = math.sqrt((uav.x - self.x)**2 + (uav.y - self.y)**2)
                #自由空间的路径损耗
                h_free = 20 * math.log(4 * math.pi * self.fc * d_uav / self.c , 2)

                if d_uav < 0.05:
                    reward -= 3
                elif d_uav < 0.05 + 2 * self.stepWay and d_uav >= 0.05:
                    reward -= 2
                elif h_free > self.h_free_up:
                    reward -= 0.05

        if d_tar_next < (2 * self.stepWay):
            self.done = True
            #reward += 1

        reward += reward_tar + reward_tu + reward_ob
        # print('reward: %.2f\treward_tar: %.2f\treward_tu: %.2f\treward_ob:%.2f'%(reward, reward_tar, reward_tu, reward_ob))

        self.d_tar = d_tar_next
        self.dir_tar = math.atan2(self.y_tar - self.y, self.x_tar - self.x)
        self.get_tu_can()

        return reward, self.done

