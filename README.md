# ppo_path_planning
基于ppo的路径规划，便于编写环境，ppo算法源于https://github.com/tinyzqh/light_mappo  
在mappo中将智能体数量设置为1

## 环境
  强化学习的environment与agent交互部分在env_core.py中，包括智能体的运动学模型、地图信息、奖励函数等。
  '''
      def __init__(self):
        self.agent_num = 1  # 设置智能体的个数
        self.obs_dim = 4  # 设置智能体的观测纬度
        self.action_dim = 1  # 设置智能体的动作纬度
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
'''

## 地图
  地图在envs/map.py，包括地图边界与障碍物，以线段的形式表示。在map.py中将线段储存为三维列表，在env_core.py中初始化。
  
## 输出图片
  reward、loss等强化学习曲线在result中通过tensorboard --logdir xxx命令查看。
  运动轨迹图通过envs/env_runner.py输出，图片保存在train目录和train/picture目录下。
  
  默认环境训练四百多轮可以实现这样的效果
  
![轨迹图](https://github.com/m1ntzz/ppo_path_planning/assets/102210809/d82eda3c-7ed5-4f42-aac9-0466c8fabf37)
![奖励曲线](https://github.com/m1ntzz/ppo_path_planning/assets/102210809/cc53ee7d-7716-4891-8ff0-e4dc2c8d59fa)

## 调参
  在config.py中调参，效果不好先找环境bug，再调整奖励函数，最后考虑调参
