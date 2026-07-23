import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from TerminalUser import TerminalUser
from UAV import UAV
from obstacle import obstacle
from obstacle import obstacle_dynamic


def fix_theta(theta, down, up):
    t = theta
    d = up - down
    t += d if t < down else 0
    t -= d if t > up else 0
    return t


def draw_cylinder(x, y, h, r, resolution, ax):
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, h, resolution)
    theta, Z = np.meshgrid(theta, z)

    X = r * np.cos(theta) + x
    Y = r * np.sin(theta) + y
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color='k', alpha=0.4)

    z_top = h
    z_bottom = 0
    theta_disc = np.linspace(0, 2 * np.pi, resolution)
    X_disc = r * np.cos(theta_disc) + x
    Y_disc = r * np.sin(theta_disc) + y

    ax.plot_trisurf(X_disc, Y_disc, [z_top] * resolution, color='k', alpha=0.75)
    ax.plot_trisurf(X_disc, Y_disc, [z_bottom] * resolution, color='k', alpha=0.4)


class ENV:
    def __init__(self):
        self.length = 1000
        self.width = 1000
        self.x_bound = 1
        self.y_bound = 1
        self.map_size_m = 1000
        self.map_area_km2 = 1.0

        self.n_uav = 3
        self.n_tus = 30
        self.n_obs = 20
        self.n_ob_dynamic = 5
        self.num_actions = 10 + 1
        self.num_tu_can = 5
        self.tu_allocation = []
        self.tu_service_state = []
        self.maxI = 5
        self.num_lidar = 16
        self.maxr = 0.07
        self.target = []
        self.tus = []
        self.obs = []
        self.obs_dynamic = []
        self.theta_ob_avg = 0
        self.uavs = []
        self.event_types = (
            ('rescue_call', 5, 0.030, 0.050),
            ('hazard_front', 4, 0.024, 0.048),
            ('road_block', 4, 0.020, 0.045),
            ('supply_request', 3, 0.016, 0.042),
            ('status_report', 3, 0.012, 0.040),
        )
        self.train_tu_range = (18, 36)
        self.test_tu_count = 30
        self.max_event_spawn_step = 60
        self.time_step = 0
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.collaboration = True

    def seed(self, seed=None):
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

    def _is_in_road_corridor(self, x, y):
        main_vertical = 0.46 <= x <= 0.54
        main_horizontal = 0.40 <= y <= 0.50
        branch_corridor = 0.18 <= x <= 0.74 and 0.62 <= y <= 0.70
        return main_vertical or main_horizontal or branch_corridor

    def _is_position_valid(self, x, y, min_obstacle_gap=0.0, min_tu_gap=0.0):
        if x < 0.03 or x > 0.97 or y < 0.03 or y > 0.97:
            return False
        for ob in self.obs:
            if math.sqrt((x - ob.x) ** 2 + (y - ob.y) ** 2) < ob.r + min_obstacle_gap:
                return False
        for tu in self.tus:
            if math.sqrt((x - tu.x) ** 2 + (y - tu.y) ** 2) < min_tu_gap:
                return False
        return True

    def _generate_village_obstacles(self):
        self.obs = []
        cluster_specs = [
            ((0.24, 0.24), 5),
            ((0.76, 0.26), 5),
            ((0.28, 0.78), 5),
            ((0.76, 0.76), 3),
        ]
        edge_specs = [(0.14, 0.56), (0.86, 0.58)]
        min_center_gap = 0.11

        for (cx, cy), count in cluster_specs:
            generated = 0
            attempts = 0
            while generated < count and attempts < 500:
                attempts += 1
                ob_r = random.uniform(0.032, 0.055)
                ob_x = min(max(random.gauss(cx, 0.055), 0.08), 0.92)
                ob_y = min(max(random.gauss(cy, 0.055), 0.08), 0.92)
                if self._is_in_road_corridor(ob_x, ob_y):
                    continue
                if self._is_far_enough_from_obstacles(ob_x, ob_y, ob_r, min_center_gap):
                    self.obs.append(obstacle(ob_x, ob_y, ob_r))
                    generated += 1

        for cx, cy in edge_specs:
            attempts = 0
            while attempts < 250:
                attempts += 1
                ob_r = random.uniform(0.030, 0.045)
                ob_x = min(max(random.gauss(cx, 0.030), 0.08), 0.92)
                ob_y = min(max(random.gauss(cy, 0.040), 0.08), 0.92)
                if self._is_in_road_corridor(ob_x, ob_y):
                    continue
                if self._is_far_enough_from_obstacles(ob_x, ob_y, ob_r, min_center_gap):
                    self.obs.append(obstacle(ob_x, ob_y, ob_r))
                    break

        while len(self.obs) < self.n_obs:
            ob_r = random.uniform(0.030, 0.050)
            ob_x = random.uniform(0.08, 0.92)
            ob_y = random.uniform(0.08, 0.92)
            if self._is_in_road_corridor(ob_x, ob_y):
                continue
            if self._is_far_enough_from_obstacles(ob_x, ob_y, ob_r, min_center_gap):
                self.obs.append(obstacle(ob_x, ob_y, ob_r))

    def _is_far_enough_from_obstacles(self, x, y, radius, min_center_gap):
        for ob in self.obs:
            min_gap = max(min_center_gap, ob.r + radius + 0.02)
            if math.sqrt((x - ob.x) ** 2 + (y - ob.y) ** 2) < min_gap:
                return False
        return True

    def _sample_event_position(self, flag_vertical):
        attempts = 0
        while attempts < 600:
            attempts += 1
            if random.random() < 0.65:
                if random.random() < 0.5:
                    x = random.uniform(0.12, 0.88)
                    y = random.uniform(0.34, 0.56)
                else:
                    x = random.uniform(0.42, 0.58)
                    y = random.uniform(0.12, 0.88)
            else:
                if flag_vertical:
                    x = random.uniform(0.06, 0.94)
                    y = random.uniform(0.16, 0.84)
                else:
                    x = random.uniform(0.16, 0.84)
                    y = random.uniform(0.06, 0.94)
            if self._is_position_valid(x, y, min_obstacle_gap=0.06, min_tu_gap=0.06):
                return x, y
        raise RuntimeError('Failed to sample a valid event position.')

    def _configure_event(self, tu, flag_test):
        event_type, priority, decay_rate, radius = random.choice(self.event_types)
        tu.event_type = event_type
        tu.event_priority = priority
        tu.decay_rate = decay_rate
        tu.r = radius
        max_spawn = self.max_event_spawn_step if not flag_test else self.max_event_spawn_step // 2
        tu.spawn_time = random.randint(0, max_spawn)
        tu.active = tu.spawn_time == 0
        tu.last_decay_step = 0
        tu.I = float(priority)
        tu.I_origin = float(priority)

    def _generate_event_terminals(self, flag_test, flag_vertical):
        self.tu_x = []
        self.tu_y = []
        self.tus = []
        n_tus = self.test_tu_count if flag_test else np.random.randint(*self.train_tu_range)
        self.n_tus = n_tus

        for i in range(n_tus):
            tu_x, tu_y = self._sample_event_position(flag_vertical)
            tu = TerminalUser(i, tu_x, tu_y, self.maxI)
            self._configure_event(tu, flag_test)
            self.tus.append(tu)
            self.tu_x.append(self.length - tu_x * self.length)
            self.tu_y.append(tu_y * self.width)

    def _update_event_states(self):
        for tu in self.tus:
            if tu.flag_done:
                continue
            if not tu.active and self.time_step >= tu.spawn_time:
                tu.active = True
            if not tu.active:
                continue
            elapsed_steps = self.time_step - tu.last_decay_step
            if elapsed_steps <= 0:
                continue
            tu.I = max(0.5, tu.I - tu.decay_rate * elapsed_steps)
            tu.last_decay_step = self.time_step

    def reset(self, flag_test=False):
        flag_vertical = True
        flag_up_right = True
        self.time_step = 0

        self._generate_village_obstacles()
        self._generate_event_terminals(flag_test, flag_vertical)
        self._update_event_states()

        self.d_ob_x = []
        self.d_ob_y = []
        self.d_ob_z = []
        self.d_ob_x_init = []
        self.d_ob_y_init = []
        self.d_ob_z_init = []

        n_ob_dy = self.n_ob_dynamic if flag_test else np.random.randint(2, 9)
        self.obs_dynamic = []
        theta_temp = 0
        for i in range(n_ob_dy):
            while True:
                ob_x = random.uniform(0.15, 0.85 * self.x_bound)
                ob_y = random.uniform(0.15, 0.85 * self.y_bound)
                flag = True
                for ob in self.obs:
                    if math.sqrt((ob_x - ob.x) ** 2 + (ob_y - ob.y) ** 2) < ob.r + 0.08:
                        flag = False
                        break
                if not flag:
                    continue
                for ob_d in self.obs_dynamic:
                    if math.sqrt((ob_x - ob_d.x) ** 2 + (ob_y - ob_d.y) ** 2) < 0.175:
                        flag = False
                        break
                if flag:
                    break
            self.obs_dynamic.append(obstacle_dynamic(self, ob_x, ob_y, 0.05))
            theta_temp += self.obs_dynamic[i].theta
            self.d_ob_x.append([])
            self.d_ob_y.append([])
            self.d_ob_z.append([])
            self.d_ob_x_init.append((1 - ob_x) * self.length)
            self.d_ob_y_init.append(ob_y * self.length)
            self.d_ob_z_init.append(100)
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic

        self.target = []
        self.target_x = []
        self.target_y = []
        for i in range(self.n_uav):
            if flag_vertical:
                tar_x = 0.5 * self.x_bound
                if flag_up_right:
                    tar_y = random.uniform(0.95 * self.y_bound, 1 * self.y_bound)
                else:
                    tar_y = random.uniform(0 * self.y_bound, 0.05 * self.y_bound)
            else:
                tar_y = (0.5 + i) / self.n_uav * self.y_bound
                if flag_up_right:
                    tar_x = random.uniform(0.95 * self.y_bound, 1 * self.x_bound)
                else:
                    tar_x = random.uniform(0 * self.y_bound, 0.05 * self.x_bound)
            self.target.append([tar_x, tar_y])
            self.target_x.append(tar_x * self.length)
            self.target_y.append(tar_y * self.width)

        self.uavs = []
        if flag_vertical:
            uav_x = random.uniform(0.4 * self.x_bound, 0.6 * self.x_bound)
            if flag_up_right:
                uav_y = random.uniform(0, 0.02)
            else:
                uav_y = random.uniform(0.92 * self.y_bound, 0.98 * self.y_bound)
        else:
            uav_y = random.uniform(0.4 * self.y_bound, 0.6 * self.y_bound)
            if flag_up_right:
                uav_x = random.uniform(0, 0.02)
            else:
                uav_x = random.uniform(0.92 * self.x_bound, 0.98 * self.x_bound)
        for i in range(self.n_uav):
            if flag_vertical:
                self.uavs.append(UAV(self, i, uav_x + (i - self.n_uav // 2) * 0.05, uav_y, self.target[i]))
            else:
                self.uavs.append(UAV(self, i, uav_x, uav_y + (i - self.n_uav // 2) * 0.05, self.target[i]))

        self.tu_service_state = []
        uav_state = []
        near_uav_id = []
        for uav in self.uavs:
            state, near_id = uav.state()
            uav_state.append(state)
            near_uav_id.append(near_id)
            self.tu_service_state.append(0)
        return uav_state, near_uav_id

    def render(self, flag=0):
        if flag:
            plt.clf()
            plt.xlim(-0.05 * self.length * self.x_bound, 1.05 * self.length * self.x_bound)
            plt.ylim(-0.05 * self.width * self.y_bound, 1.05 * self.width * self.y_bound)
            for uav in self.uavs:
                plt.scatter(uav.x_tar * self.length, uav.y_tar * self.width, s=20, color='r', marker='x')
            for ob in self.obs:
                theta = np.arange(0, 2 * np.pi, 0.01)
                x = ob.x * self.length + ob.r * self.length * np.cos(theta)
                y = ob.y * self.width + ob.r * self.length * np.sin(theta)
                plt.fill(x, y, ob.r, color='black')

        for ob_d in self.obs_dynamic:
            plt.scatter(ob_d.x * self.length, ob_d.y * self.width, s=10, color='black', marker='^')
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = ob_d.x * self.length + ob_d.r * self.length * np.cos(theta)
            y = ob_d.y * self.width + ob_d.r * self.length * np.sin(theta)
            plt.fill(x, y, ob_d.r, color='gray')

        for tu in self.tus:
            if not tu.active:
                plt.scatter(tu.x * self.length, tu.y * self.width, s=10, color='lightgray', marker='x')
                continue
            plt.scatter(tu.x * self.length, tu.y * self.width, s=10, color='green' if tu.flag_done else 'brown', marker='*')
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = tu.x * self.length + tu.r * self.length * np.cos(theta)
            y = tu.y * self.width + tu.r * self.length * np.sin(theta)
            plt.plot(x, y, color='green' if tu.flag_done else 'brown')
            plt.text((tu.x + 0.01) * self.length, (tu.y + 0.01) * self.width, '%.2f' % tu.I)

        for uav in self.uavs:
            plt.scatter(uav.x * self.length, uav.y * self.width, s=10, color='b', marker='^')
        plt.pause(0.01)

    def render_3D(self, uav_x, uav_y, uav_z):
        fig = plt.figure(figsize=(10, 9))
        ax = Axes3D(fig)
        fig.add_axes(ax)

        for ob in self.obs:
            draw_cylinder(ob.y * self.length, (1 - ob.x) * self.length, 200, ob.r * self.length, 100, ax)

        ax.scatter3D(self.tu_y, self.tu_x, np.ones(len(self.tus)) * 0, color='green', marker='^', s=80, alpha=1, zorder=10)
        ax.scatter3D(self.d_ob_y_init, self.d_ob_x_init, self.d_ob_z_init, color='red', marker='o', s=30, alpha=1, linewidths=5, zorder=10)
        for i in range(self.n_ob_dynamic):
            ax.plot3D(self.d_ob_y[i], self.d_ob_x[i], self.d_ob_z[i], 'r-.', zorder=10)
        ax.plot3D(uav_x[0], uav_y[0], uav_z[0], c='blue', label='UAV_0', zorder=10)
        ax.plot3D(uav_x[1], uav_y[1], uav_z[1], c='blue', label='UAV_1', zorder=10)
        ax.plot3D(uav_x[2], uav_y[2], uav_z[2], c='blue', label='UAV_2', zorder=10)
        ax.view_init(elev=65, azim=180)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

    def step(self, action):
        self.time_step += 1
        self._update_event_states()

        theta_temp = 0
        for i, ob_d in enumerate(self.obs_dynamic):
            ob_d.update()
            self.d_ob_x[i].append((1 - ob_d.x) * self.length)
            self.d_ob_y[i].append(ob_d.y * self.length)
            self.d_ob_z[i].append(100)
            theta_temp += ob_d.theta
        if self.n_ob_dynamic > 0:
            self.theta_ob_avg = theta_temp / self.n_ob_dynamic

        reward = []
        next_state = []
        near_uav_id = []
        done = []
        hover_flag = []
        for _ in range(self.n_uav):
            done.append(True)
            hover_flag.append(False)
        done = np.array(done, dtype=bool)

        for uav in self.uavs:
            if uav.done:
                continue
            for tu in self.tus:
                if tu.flag_done or not tu.active:
                    continue
                d = math.sqrt((uav.x - tu.x) ** 2 + (uav.y - tu.y) ** 2)
                if d < tu.r:
                    hover_flag[uav.id] = True
                    dir_tu = math.sqrt((uav.x - tu.x) ** 2 + (uav.x - uav.y) ** 2 + uav.H ** 2)
                    theta_tu = np.arcsin(uav.H / dir_tu)
                    theta_tu = fix_theta(theta_tu, -1, 1)
                    uav.pro = 1 / (1 + uav.alpa * math.exp(-uav.beta * (theta_tu - uav.alpa)))
                    uav.h = uav.K0 * dir_tu ** 2 * (uav.pro * uav.mu_los + (1 - uav.pro) * uav.mu_nlos)
                    S = uav.P / (uav.noise * uav.h)
                    rate_tu = uav.B * math.log(1 + S, 2)
                    R_max = uav.B * math.log((1 + uav.P / (uav.noise * uav.K0 * uav.H * uav.H * uav.mu_los)), 2)
                    rate_tu = rate_tu / R_max
                    if tu.I > 0:
                        tu.I -= rate_tu
                    if tu.I <= 0:
                        tu.I = 0
                        tu.flag_done = True
                        uav.iot_service_cnt += 1
                        self.tu_service_state[uav.id] += 1
                        hover_flag[uav.id] = False

        for i in range(self.n_uav):
            if self.uavs[i].done:
                reward.append(0)
                next_state.append([])
                near_uav_id.append([])
                continue
            reward_t, done_t = self.uavs[i].update(action[i], hover_flag[i])
            reward.append(reward_t)
            if done_t is False:
                done[i] = False
            state, near_id = self.uavs[i].state()
            next_state.append(state)
            near_uav_id.append(near_id)

        if self.collaboration:
            iot_state = []
            for tu in self.tus:
                if tu.flag_done or not tu.active:
                    continue
                covered = False
                for id_s in range(self.n_uav):
                    if self.uavs[id_s].done:
                        continue
                    d = math.sqrt((self.uavs[id_s].x - tu.x) ** 2 + (self.uavs[id_s].y - tu.y) ** 2)
                    if d < self.uavs[id_s].senser_tu_r:
                        covered = True
                        break
                iot_state.append(1 if covered else 0)
            numerator = 0
            denominator = 0
            for val in iot_state:
                numerator += val
                denominator += val ** 2
            if denominator != 0 and len(iot_state) > 0:
                fairness_iot = numerator ** 2 / denominator / len(iot_state)
            else:
                fairness_iot = 0

            cnt = 0
            numerator = 0
            denominator = 0
            for uav in self.uavs:
                if uav.done and uav.iot_service_rwd >= 0:
                    numerator += uav.iot_service_rwd
                    denominator += uav.iot_service_rwd ** 2
                    cnt += 1
            if denominator != 0 and cnt != 0:
                fairness_uav = numerator ** 2 / denominator / cnt
            else:
                fairness_uav = 0

            for uav in self.uavs:
                if uav.done:
                    reward[uav.id] += fairness_iot + fairness_uav

        return next_state, reward, done, near_uav_id


if __name__ == '__main__':
    env = ENV()
    env.reset(flag_test=True)
