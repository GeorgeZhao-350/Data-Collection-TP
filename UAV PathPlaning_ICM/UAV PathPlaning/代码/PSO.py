import numpy as np
import math

class PSO:
    def __init__(self, n_uav, uav_pos, tu_pos, n_tus, end_pos):  # n_uav <int>, uav_pos <list>, tu_pos <list>, n_tus <int>
        self.tu_pos=tu_pos
        self.tu_num=n_tus
        self.uav_num=n_uav
        self.max_iteration=60

        self.UAV_start = uav_pos
        self.target_pos = tu_pos
        self.row = n_uav
        self.column = n_tus
        self.end_pos=end_pos

        self.UtoT, self.TtoT, self.TtoE, self.max_dis = self.dist_normal()
        self.disU = [[] for k in range(self.uav_num)]  # 存储每个无人机飞过的每段距离

        # 后期加速因子和惯性权重采用微分递减修改
        self.w = 1
        self.c1 = self.c2 = 2
        self.population_size = 100  # 粒子群数量
        self.dim = n_tus  # 搜索空间的维度
        # self.max_steps = globalv.max_steps   # 迭代次数
        self.x_bound = [0, n_uav - 0.01]  # 解空间范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # 初始化粒子群位置
        self.v = np.random.uniform(-1, 1, (self.population_size, self.dim))  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度
        # 存储时间与与之对应的适应值
        self.time_mat = []
        self.fitness_mat = []

    def allocation(self):
        iteration = 0  # 存储程序当前运行时间
        while (iteration < self.max_iteration):
            iteration += 1
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.v = np.clip(self.v, -1, 1)  # 限定速度范围
            self.x = self.v + self.x
            self.x = np.clip(self.x, self.x_bound[0], self.x_bound[1])
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            self.time_mat.append(iteration)
            self.fitness_mat.append(self.global_best_fitness)
            # print('当前运行时间：%.5f' % (iteration))
            # print('当前最优值: %.5f' % (self.global_best_fitness))
        print('Best分配方案：', self.decode(self.pg))  # 显示任务分配结果 # self.x[np.argmin(fitness)]
        scheme = self.decode(self.pg)  # self.x[np.argmin(fitness)]
        return scheme

    def decode(self, taskcode):
        UandT = [[] for k in range(self.uav_num)]  # 存储每架无人机需要执行的任务
        # 为无人机分配任务
        for i in range(self.tu_num):
            detected = 0
            for j in range(self.uav_num):
                if int(taskcode[i]) == j:
                    detected = 1
                    a = 0  # 设置标志位判断当前值是否为列表中第一个元素
                    for l in range(len(UandT[j])):
                        if taskcode[i] < taskcode[l]:
                            UandT[j].insert(l, i)  # 为任务进行排序
                            a = 1
                            break
                    if a != 1:
                        UandT[j].append(i)
            if detected == 0:
                print("ERROR: IoT node allocation error!")
        return UandT

    def calculate_fitness(self, taskcode):
        fitness = np.array([])
        for k in range(self.population_size):
            cost_sum = 0
            avgcost = []
            UandT = self.decode(taskcode[k])  # 每架无人机分配了哪些任务
            for i in range(self.uav_num):
                UAV = UandT[i]
                cost = 0
                if len(UAV) == 0:  # 无人机没有任务
                    pass
                elif len(UAV) == 1:  # 无人机有一个任务
                    dis0 = self.UtoT[i][UAV[0]]
                    dis1 = self.TtoE[i][UAV[0]]
                    self.disU[i].extend([dis0, dis1])
                else:  # 无人机有超过一个任务
                    dis1 = self.UtoT[i][UAV[0]]
                    self.disU[i].append(dis1)
                    for j in range(len(UAV) - 1):
                        dis2 = self.TtoT[UAV[j]][UAV[j + 1]]
                        self.disU[i].append(dis2)
                    dis3 = self.TtoE[i][UAV[len(UAV) - 1]]
                    self.disU[i].append(dis3)
                cost = sum(self.disU[i])
                # cost = cost + cont(cost*self.max_dis)
                cost_sum = cost_sum + cost
                avgcost.append(cost)
            fitness = np.append(fitness, max(avgcost))
        return fitness

    def dis_UT(self):
        UtoT = np.zeros((self.row, self.column))  # 初始化矩阵信息
        for i in range(self.row):
            for j in range(self.column):
                d_x = self.UAV_start[i][0] - self.target_pos[j][0]
                d_y = self.UAV_start[i][1] - self.target_pos[j][1]
                UtoT[i][j] = math.sqrt(d_x ** 2 + d_y ** 2)
        return UtoT  # 函数返回距离矩阵

    # 各目标点之间的距离
    def dis_TT(self):
        TtoT = np.zeros((self.column, self.column))
        for i in range(self.column):
            for j in range(self.column):
                d_x = self.target_pos[i][0] - self.target_pos[j][0]
                d_y = self.target_pos[i][1] - self.target_pos[j][1]
                TtoT[i][j] = math.sqrt(d_x ** 2 + d_y ** 2)
        return TtoT

    # target to end point
    def dis_TE(self):
        TtoE = np.zeros((self.row, self.column))
        for i in range(self.row):
            for j in range(self.column):
                d_x = self.end_pos[i][0] - self.target_pos[j][0]
                d_y = self.end_pos[i][1] - self.target_pos[j][1]
                TtoE[i][j] = math.sqrt(d_x ** 2 + d_y ** 2)
        return TtoE

        # 距离归一化
    def dist_normal(self):
        UtoT = self.dis_UT()
        TtoT = self.dis_TT()
        TtoE = self.dis_TE()
        max_dis = max(UtoT.max(), TtoT.max(), TtoE.max())
        UtoT = UtoT / max_dis
        TtoT = TtoT / max_dis
        TtoE = TtoT / max_dis
        return UtoT, TtoT, TtoE, max_dis
