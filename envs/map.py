class Map:
    def __init__(self):
        self.x_range = 50  # size of background
        self.y_range = 30

    def limit(self):

        x = self.x_range
        y = self.y_range

        # 地图边界
        margin = [[[0, 0], [x, 0]], [[x, 0], [x, y]], [[x, y], [0, y]], [[0, 0], [0, y]]]
        # 障碍物（线段）
        barriers = [[[10, 15], [20, 15]],
                    [[20, 15], [20, 0]],
                    [[28, 15], [28, 30]],
                    [[40, 0], [40, 15]]]

        margin.extend(barriers)
        map_limit = margin
        return map_limit
