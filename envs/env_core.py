import numpy as np
import math
from envs import map


class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 1  # 设置智能体的个数
        self.obs_dim = 4  # 设置智能体的观测纬度
        self.action_dim = 1  # 设置智能体的动作纬度，这里假定为一个五个纬度的
        self.u_max = math.pi / 3  # 最大航向角速度
        self.delta_t = 0.2  # 仿真步长
        self.v = 4  # 巡航速度
        self.agent_size = 1  # 智能体半径
        self.safe_distance = 2  # 安全距离
        self.reward_co = [1, 8, 12, 15]  # reward_coefficient:奖励系数

        self.u = 0  # 航向角速度
        self.time = 0  # 时间
        self.done = [False]  # 是否完成
        self.t_ach = [300]  # 完成的时间

        self.states = []

        # 加载地图（线段）
        self.map1 = map.Map()
        self.map_limit = self.map1.limit()

        self.target = [45, 25]  # 目标点位置

    def reset(self):

        self.time = 0
        self.u = 0
        self.states = []
        self.done = [False]
        self.t_ach = [300]
        self.states = [[5, 5, 0, 3]]
        return self.states

    def step(self, actions):

        self.time += self.delta_t  # 计算当前时间
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):

            # 初始化奖励
            r0 = 0
            r1 = 0
            r2 = 0
            r3 = 0

            # 更新智能体的位置和角度
            self.u += actions[i, 0]
            if self.u >= self.u_max:
                self.u = self.u_max
            elif self.u <= -self.u_max:
                self.u = -self.u_max
            self.states[i][2] += self.u * self.delta_t
            self.states[i][0] += self.v * math.cos(self.states[i][2]) * self.delta_t
            self.states[i][1] += self.v * math.sin(self.states[i][2]) * self.delta_t

            # 智能体与目标点的距离
            x_t = np.sqrt((self.states[i][0] - self.target[0]) ** 2 + (self.states[i][1] - self.target[1]) ** 2)

            # 智能体与障碍的距离
            x_b = []
            for line in range(len(self.map_limit)):
                dis = self.dis_point_to_seg_line([self.states[i][0], self.states[i][1]], self.map_limit[line][0], self.map_limit[line][1])
                if dis > self.agent_size + self.safe_distance:
                    dis = self.agent_size + self.safe_distance
                x_b.append(dis)

            # 引导奖励
            r0 = - self.time
            r1 = - x_t

            # 避撞奖励
            if min(x_b) < 1:
                r2 = -300

            # 计算是否完成
            if self.t_ach[i] < self.time:
                self.done[i] = True

            # 结算奖励
            if x_t < 1:
                self.t_ach[i] = self.time
                r3 = 200
            else:
                self.done[i] = False

            r = self.reward_co[0]*r0 + self.reward_co[1]*r1 + self.reward_co[2]*r2 + self.reward_co[3]*r3

            sub_agent_reward.append([r])

            sub_agent_done.append(self.done[i])

            sub_agent_info.append({})

        return [self.states, sub_agent_reward, sub_agent_done, sub_agent_info]

    @staticmethod
    # 计算点到线段的距离
    def dis_point_to_seg_line(p, a, b):
        a, b, p = np.array(a), np.array(b), np.array(p)  # trans to np.array
        d = np.divide(b - a, np.linalg.norm(b - a))  # normalized tangent vector
        s = np.dot(a - p, d)  # signed parallel distance components
        t = np.dot(p - b, d)
        h = np.maximum.reduce([s, t, 0])  # clamped parallel distance
        c = np.cross(p - a, d)  # perpendicular distance component
        return np.hypot(h, np.linalg.norm(c))
