import numpy as np
import sys
import math
sys.path.append("..")
from train import train1


class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 3  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 3  # 设置智能体的观测纬度
        self.action_dim = 1  # 设置智能体的动作纬度，这里假定为一个五个纬度的
        self.u_max = math.pi / 4  # 最大航向角速度
        self.delta_t = 0.1  # 仿真步长
        self.v = 15  # 巡航速度
        self.agent_size = 3  # 无人机半径
        self.safe_distance = 10  # 安全飞行距离
        self.reward_co = [1, 2, 6, 24]  # reward_coefficient:奖励系数
        self.f_range = [40, 40, 75]

        self.u = [0., 0., 0.]  # 航向角速度
        self.time = 0  # 时间
        self.done = [False, False, False]  # 是否完成
        self.tar_ach = [False, False, False]  # target_achieved:已完成的目标点
        self.t_ach = [300, 300, 300]

        self.states = []

        # 一定范围内随机生成目标点
        # self.target_x = np.random.randint(40, 200, (3, 1))
        # self.target_y = np.random.randint(10, 190, (3, 1))
        # self.target = np.append(self.target_x, self.target_y, axis=1)
        self.target = np.array([[200, 250], [230, 175], [270, 25]])

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        self.time = 0
        self.u = [0., 0., 0.]
        self.states = []
        self.done = [False, False, False]
        self.tar_ach = [False, False, False]
        self.t_ach = [300, 300, 300]
        for i in range(0, 3):
            self.states.append([20, 50*i + 50, 0])
        return self.states

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        self.time += self.delta_t  # 计算当前时间
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):

            if train1.main(sys.argv[1:]).runner.run().episode <= 3650 and self.done[i]:
                sub_agent_reward.append([0])
                sub_agent_done.append(self.done[i])
                sub_agent_info.append({})
                continue

            if train1.main(sys.argv[1:]).runner.run().episode > 3650 and i == 0 and self.done[i]:
                sub_agent_reward.append([0])
                sub_agent_done.append(self.done[i])
                sub_agent_info.append({})
                continue

            if train1.main(sys.argv[1:]).runner.run().episode > 3650 and i == 2 and self.time > 4:
                sub_agent_reward.append([0])
                sub_agent_done.append(self.done[i])
                sub_agent_info.append({})
                continue

            r0 = 0
            r1 = 0
            r2 = 0
            r3 = 0
            x = [0, 0, 0]
            y = [0, 0, 0]

            # 更新智能体的位置和角度
            self.u[i] += actions[i, 0]
            if self.u[i] >= self.u_max:
                self.u[i] = self.u_max
            elif self.u[i] <= -self.u_max:
                self.u[i] = -self.u_max
            self.states[i][2] += self.u[i] * self.delta_t
            self.states[i][0] += self.v * math.cos(self.states[i][2]) * self.delta_t
            self.states[i][1] += self.v * math.sin(self.states[i][2]) * self.delta_t

            # 计算智能体与目标点的距离
            x[0] = np.sqrt((self.states[i][0] - self.target[0, 0]) ** 2 + (self.states[i][1] - self.target[0, 1]) ** 2)
            x[1] = np.sqrt((self.states[i][0] - self.target[1, 0]) ** 2 + (self.states[i][1] - self.target[1, 1]) ** 2)
            x[2] = np.sqrt((self.states[i][0] - self.target[2, 0]) ** 2 + (self.states[i][1] - self.target[2, 1]) ** 2)

            for num_target in range(0, 3):
                if not self.tar_ach[num_target]:
                    # r0 -= - 4000 / pow(x[num_target], 2) + 45
                    r0 -= self.field(x[num_target], self.f_range[num_target])

            for n in range(0, 3):
                if i != n and self.done[n] is False:
                    d = np.sqrt((self.states[i][0] - self.states[n][0]) ** 2 + (self.states[i][1] - self.states[n][1]) ** 2)
                    if d < 2 * self.agent_size + self.safe_distance:
                        r2 -= (2 * self.agent_size + self.safe_distance - d) ** 2

            # 计算是否完成
            if self.t_ach[i] < self.time and i != 1:
                self.done[i] = True
            if min(x) < 10:
                if not self.tar_ach[np.argmin(x)]:
                    self.tar_ach[np.argmin(x)] = True
                    self.t_ach[i] = self.time
                    r3 = 800
            else:
                self.done[i] = False

            r = self.reward_co[0]*r0+self.reward_co[1]*r1+self.reward_co[2]*r2+self.reward_co[3]*r3

            sub_agent_reward.append([r])

            sub_agent_done.append(self.done[i])

            sub_agent_info.append({})

        return [self.states, sub_agent_reward, sub_agent_done, sub_agent_info]

    @staticmethod
    def hermite(x0, x1, y0, y1, y0_prime, y1_prime, x):
        alpha0 = lambda x: ((x - x1) / (x0 - x1)) ** 2 * (2 * (x - x0) / (x1 - x0) + 1)
        alpha1 = lambda x: ((x - x0) / (x1 - x0)) ** 2 * (2 * (x - x1) / (x0 - x1) + 1)
        beta0 = lambda x: ((x - x1) / (x0 - x1)) ** 2 * (x - x0)
        beta1 = lambda x: ((x - x0) / (x1 - x0)) ** 2 * (x - x1)
        H = alpha0(x) * y0 + alpha1(x) * y1 + beta0(x) * y0_prime + beta1(x) * y1_prime
        return H

    @staticmethod
    def field(x, f_range):
        f_splite = (f_range - 9) / 3
        if x < f_splite:
            return EnvCore.hermite(9, f_splite, 0, 40, 7, 2.5, x)
        elif f_splite < x < f_range:
            return EnvCore.hermite(f_splite, f_range, 40, 80, 2.5, 0.1, x)
        else:
            return 0.1 * x + 80 - 0.1 * f_range

    @staticmethod
    def angle_of_vector(v1, v2):
        pi = 3.141592
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = np.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * np.sqrt(pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        if v2[1] < 0:
            return -np.arccos(cos)
        else:
            return np.arccos(cos)
