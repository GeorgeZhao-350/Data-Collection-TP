import math
import numpy as np

def fix_theta(theta, down, up):
    t = theta
    d = up - down
    t += d if t < down else 0
    t -= d if t > up else 0
    return t

alpa = 10  #LOS概率控制量
beta = 0.6
fc = 2.4e9 #载波频率
c = 3e8 #光速
K0 = 16 * (math.pi ** 2) * (fc ** 2) /(c ** 2) #路径损耗系数K0
mu_los = 1.45 #los链路衰减因子
mu_nlos = 200 #nlos链路衰减因子
noise = pow(10,-15) #噪声功率为-90dbm
B = 1 #带宽
P = pow(10, -6)  # 传感器的发射功率 0.01mW,-20dbm

deg = 80/180*math.pi
H = 100
d = 50

dist = math.sqrt(d**2 + H**2)

theta_tu_high = np.arcsin(H / dist)
# theta_tu_high = fix_theta(theta_tu_high, -1, 1)  # 归一化
theta_tu_high = theta_tu_high/math.pi*180
# 视距概率
pro = 1 / (1 + alpa * math.exp(-beta * (theta_tu_high - alpa)))
# 路径损耗
h = K0 * dist ** 2 * (pro * mu_los + (1 - pro) * mu_nlos)
# 数据采集速率
S = P / (noise * h)
S_db = 20*math.log(S, 10)
rate = B * math.log(1 + S, 2)

print("SNR: %.3f" %S_db)
print("Rate: %.3f" %rate)

d_uav = 500

h_free = (4 * math.pi * fc * d_uav / c)**2
S = P / (noise * h_free)
S_db = 20*math.log(S, 10)
rate = B * math.log(1 + S, 2)

print("SNR(LOS): %.3f" %S_db)
print("Rate(LOS): %.3f" %rate)