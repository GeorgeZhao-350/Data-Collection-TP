import numpy as np
import random
import math

class obstacle():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r #障碍物实体范围，威胁阈值为0.8
        self.rr = r**2 / (2*math.log(1/0.8)) #威胁值分布的标准差


class obstacle_dynamic():
    def __init__(self, env, x, y, r):
        self.env = env
        self.x = x
        self.y = y
        self.r = r
        self.rr = r ** 2 / (2 * math.log(1 / 0.8))  # 威胁值分布的标准差
        self.v = 0.003     #障碍物移动速度

        self.theta = random.uniform(0, 2 * math.pi) - math.pi #生成-π 到 π 之间的随机数
        self.k = 0.98   #上一时隙的方向角对下一时隙方向角的影响
        self.eps_theta = 0  # 方向角随机部分的均值
        self.delt_theta = 1  # 方向角随机部分的标准差

    def update(self):
        x_next = self.x + self.v * math.cos(self.theta)
        y_next = self.y + self.v * math.sin(self.theta)
        theta_next = self.k * self.theta + (1 - self.k) * self.env.theta_ob_avg + \
                     math.sqrt(1 - self.k ** 2) * np.random.normal(loc=self.eps_theta, scale=self.delt_theta)
        theta_next += 2*math.pi if theta_next<-math.pi else (-2*math.pi if theta_next>math.pi else 0)
        self.x = x_next
        self.y = y_next
        self.theta = theta_next
