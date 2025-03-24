import colorsys
import hashlib
import heapq
import math
import os
import pathlib
import random
from functools import cached_property
from itertools import combinations

import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib.path import Path

from placedb import PlaceDB
from util import orient_pin_transform, orient_size_transform, trackit

ORIENT_MAP = ["N", "W", "S", "E", "FS", "FE", "FN", "FW"]

class OrientPlaceEnv(gym.Env):

    def __init__(self, placedb: PlaceDB, grid=224):
        print(f"{grid=}, {grid * grid=}")
        print(f"macro num: {len(placedb.macro_info)}")
        print(f"cell num: {len(placedb.cell_info)}")
        print(f"node num: {len(placedb.node_info)}")
        print(f"net num: {len(placedb.net_info)}")

        assert grid * grid >= len(placedb.macro_info), "grid size is too small"
        self.grid = grid
        self.canvas_width, self.canvas_height = placedb.canvas_size
        self.placedb = placedb
        self.num_macro = len(placedb.macro_info)
        self.placed_num_macro = len(placedb.macro_place_queue)
        self.num_net = len(placedb.net_info)
        self.node_name_list = placedb.macro_place_queue
        self.action_space = spaces.Discrete(self.grid * self.grid)
        self.orient_space = spaces.Discrete(8)
        state_dim = self.grid * self.grid + 8 * self.grid * self.grid + 8 * self.grid * self.grid + 3
        assert state_dim == self.FEATURE_SLICE.stop
        self.observation_space = spaces.Box(low=-10, high=10, shape=(state_dim,), dtype=np.float32)

        self.state: np.ndarray = None
        self.net_min_max_ord: dict[str, tuple[int, int, int, int]] = {}
        self.node_pos: dict[str, tuple[int, int, int, int, int]] = None
        self.node_pin_pos: dict[str, dict[str, tuple[float, float]]] = None
        self.place_counter: int = None
        self.ratio = self.canvas_height / self.grid
        print("self.ratio = {:.2f}".format(self.ratio))

        self.node_orient_physical_info: dict[str, dict[int, dict]] = {}
        self._build_orientation_physical_info()

    @cached_property
    def CANVAS_SLICE(self) -> slice:
        start = 0
        offset = self.grid * self.grid
        return slice(start, start + offset)

    @cached_property
    def WIRE_SLICE(self) -> slice:
        start = self.CANVAS_SLICE.stop
        offset = 8 * self.grid * self.grid
        return slice(start, start + offset)

    @cached_property
    def POS_SLICE(self) -> slice:
        start = self.WIRE_SLICE.stop
        offset = 8 * self.grid * self.grid
        return slice(start, start + offset)

    @cached_property
    def FEATURE_SLICE(self) -> slice:
        start = self.POS_SLICE.stop
        offset = 3
        return slice(start, start + offset)

    def _get_grid_metric(self, x: int, y: int) -> tuple[int, int]:
        grid_size_x = math.ceil(max(1, x / self.ratio))
        grid_size_y = math.ceil(max(1, y / self.ratio))
        return grid_size_x, grid_size_y

    def _build_state(
        self,
        canvas: np.ndarray,
        wire_img: np.ndarray,
        pos_mask: np.ndarray,
        place_counter: int,
        size_x: int,
        size_y: int,
    ):
        grid_size_x, grid_size_y = self._get_grid_metric(size_x, size_y)
        if wire_img.max() > 0:
            wire_img /= wire_img.max()
        return np.concatenate(
            [
                canvas.flatten(),
                wire_img.flatten(),
                pos_mask.flatten(),
                [place_counter, grid_size_x, grid_size_y],
            ],
            axis=0,
        )

    def _build_orientation_physical_info(self):
        """
        构建每个节点在不同方向上的物理信息。

        该方法首先收集所有节点的基本信息，包括尺寸和引脚位置，然后根据方向变换规则，
        计算每个节点在8个不同方向上的尺寸和引脚位置，并存储在`node_orient_physical_info`中。

        结果的schema如下：
        - node_orient_physical_info: dict[str, dict[int, dict]]
            - key: 节点名称 (str)
            - value: dict[int, dict]
                - key: 方向 (int, 0-7)
                - value: dict
                    - 'size': (width, height) 元组，表示节点在该方向上的尺寸 (tuple[float, float])
                    - 'pins': dict[str, tuple[float, float]]
                        - key: 引脚名称 (str)
                        - value: 引脚在该方向上的偏移量 (tuple[float, float])
        """
        # 初始化节点基本信息字典
        node_basic_info: dict[str, dict] = {}

        # 遍历所有节点名称，收集节点基本信息
        for node_name in self.node_name_list:
            node_basic_info[node_name] = {}
            # 获取节点的宽度和高度
            width = self.placedb.node_info[node_name]["width"]
            height = self.placedb.node_info[node_name]["height"]
            # 存储节点的尺寸信息
            node_basic_info[node_name]["size"] = (width, height)

            # 初始化节点的引脚信息字典
            node_basic_info[node_name]["pins"] = {}
            # 遍历节点所属的所有网络，收集引脚信息
            for net_name in self.placedb.node2net_dict[node_name]:
                # 一个node的多个pin可能出现在同一个net中
                for pin_name, pin_info in self.placedb.net_info[net_name]["nodes"][node_name]["pins"].items():
                    pin_offset = pin_info["pin_offset"]
                    # 存储引脚的偏移量
                    node_basic_info[node_name]["pins"][pin_name] = pin_offset

        # 遍历所有节点名称，构建每个节点在不同方向上的物理信息
        for node_name in self.node_name_list:
            self.node_orient_physical_info[node_name] = {}
            # 遍历所有可能的方向
            for orient in range(8):
                self.node_orient_physical_info[node_name][orient] = {}
                # 获取节点原始尺寸
                size = node_basic_info[node_name]["size"]
                # 计算方向变换后的尺寸
                new_size = orient_size_transform(orient, size)
                self.node_orient_physical_info[node_name][orient]["size"] = new_size

                self.node_orient_physical_info[node_name][orient]["pins"] = {}
                # 遍历节点的所有引脚，计算方向变换后的引脚位置
                for pin_name in node_basic_info[node_name]["pins"]:
                    pin_offset = node_basic_info[node_name]["pins"][pin_name]
                    # 根据方向变换引脚偏移量
                    new_pin_offset = orient_pin_transform(orient, size, pin_offset)
                    self.node_orient_physical_info[node_name][orient]["pins"][pin_name] = new_pin_offset

    def reset(self):
        canvas = np.zeros((self.grid, self.grid))
        self.node_pos = {}
        self.net_min_max_ord = {}
        self.node_pin_pos = {}
        self.rudy = np.zeros((self.grid, self.grid))
        if os.getenv("PLACEENV_IGNORE_PORT", "0") == "1":
            print("`PLACEENV_IGNORE_PORT=1`, so chip ports are ignored")
        else:
            self.node_pin_pos["PORT"] = {}
            for port_name in self.placedb.port2net_dict:
                # 加入chip端口的坐标
                self.node_pin_pos["PORT"][port_name] = tuple(self.placedb.port_info[port_name]["coordinate"])

                pin_x = round(self.placedb.port_info[port_name]["coordinate"][0] / self.ratio)
                pin_y = round(self.placedb.port_info[port_name]["coordinate"][1] / self.ratio)
                for net_name in self.placedb.port2net_dict[port_name]:
                    if net_name in self.net_min_max_ord:
                        if pin_x > self.net_min_max_ord[net_name]["max_x"]:
                            self.net_min_max_ord[net_name]["max_x"] = pin_x
                        elif pin_x < self.net_min_max_ord[net_name]["min_x"]:
                            self.net_min_max_ord[net_name]["max_y"] = pin_y
                        if pin_y > self.net_min_max_ord[net_name]["max_y"]:
                            self.net_min_max_ord[net_name]["max_y"] = pin_y
                        elif pin_y < self.net_min_max_ord[net_name]["min_y"]:
                            self.net_min_max_ord[net_name]["min_y"] = pin_y
                    else:
                        self.net_min_max_ord[net_name] = {}
                        self.net_min_max_ord[net_name]["max_x"] = pin_x
                        self.net_min_max_ord[net_name]["min_x"] = pin_x
                        self.net_min_max_ord[net_name]["max_y"] = pin_y
                        self.net_min_max_ord[net_name]["min_y"] = pin_y

        self.place_counter = 0
        node_name = self.node_name_list[self.place_counter]
        size_x, size_y = self.node_orient_physical_info[node_name][0]["size"]

        wire_img = self._get_wire_img()
        pos_mask = self._get_pos_mask()

        self.state = self._build_state(
            canvas=canvas,
            wire_img=wire_img,
            pos_mask=pos_mask,
            place_counter=self.place_counter,
            size_x=size_x,
            size_y=size_y,
        )
        return self.state

    @trackit
    def step(self, action, orient: int = 0):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        canvas = self.state[self.CANVAS_SLICE].reshape(self.grid, self.grid)
        pos_mask = self.state[self.POS_SLICE].reshape(8, self.grid, self.grid)
        mask = pos_mask[orient, :, :]

        reward = 0
        x = round(action // self.grid)
        y = round(action % self.grid)

        if mask[x][y] == 1:
            reward += -200000

        node_name = self.node_name_list[self.place_counter]
        node_size = self.node_orient_physical_info[node_name][orient]["size"]
        grid_size_x, grid_size_y = self._get_grid_metric(node_size[0], node_size[1])

        flag1 = (
            max(
                abs(grid_size_x - self.state[-2]),
                abs(grid_size_y - self.state[-1]),
            )
            < 1e-5
        )
        flag2 = (
            max(
                abs(grid_size_x - self.state[-1]),
                abs(grid_size_y - self.state[-2]),
            )
            < 1e-5
        )
        assert flag1 or flag2, f"{grid_size_x} {grid_size_y} {self.state[-2]:.2f} {self.state[-1]:.2f}"

        canvas[x : x + grid_size_x, y : y + grid_size_y] = 1.0
        canvas[x : x + grid_size_x, y] = 0.5
        if y + grid_size_y - 1 < self.grid:
            canvas[x : x + grid_size_x, max(0, y + grid_size_y - 1)] = 0.5
        canvas[x, y : y + grid_size_y] = 0.5
        if x + grid_size_x - 1 < self.grid:
            canvas[max(0, x + grid_size_x - 1), y : y + grid_size_y] = 0.5

        self.node_pos[node_name] = (x, y, grid_size_x, grid_size_y, orient)
        self.node_pin_pos[node_name] = {}

        for net_name in self.placedb.node2net_dict[node_name]:
            # 一个node的多个pin可能出现在同一个net中
            for pin_name in self.placedb.net_info[net_name]["nodes"][node_name]["pins"].keys():
                pin_offset_x, pin_offset_y = self.node_orient_physical_info[node_name][orient]["pins"][pin_name]
                self.node_pin_pos[node_name][pin_name] = (pin_offset_x, pin_offset_y)

                pin_x = round((x * self.ratio + pin_offset_x) / self.ratio)
                pin_y = round((y * self.ratio + pin_offset_y) / self.ratio)
                if net_name in self.net_min_max_ord:
                    start_x = self.net_min_max_ord[net_name]["min_x"]
                    end_x = self.net_min_max_ord[net_name]["max_x"]
                    start_y = self.net_min_max_ord[net_name]["min_y"]
                    end_y = self.net_min_max_ord[net_name]["max_y"]
                    delta_x = end_x - start_x
                    delta_y = end_y - start_y
                    if delta_x > 0 or delta_y > 0:
                        self.rudy[start_x : end_x + 1, start_y : end_y + 1] -= 1 / (delta_x + 1) + 1 / (delta_y + 1)
                    weight = 1.0
                    if "weight" in self.placedb.net_info[net_name]:
                        weight = self.placedb.net_info[net_name]["weight"]

                    if pin_x > self.net_min_max_ord[net_name]["max_x"]:
                        reward += weight * (self.net_min_max_ord[net_name]["max_x"] - pin_x)
                        self.net_min_max_ord[net_name]["max_x"] = pin_x
                    elif pin_x < self.net_min_max_ord[net_name]["min_x"]:
                        reward += weight * (pin_x - self.net_min_max_ord[net_name]["min_x"])
                        self.net_min_max_ord[net_name]["min_x"] = pin_x
                    if pin_y > self.net_min_max_ord[net_name]["max_y"]:
                        reward += weight * (self.net_min_max_ord[net_name]["max_y"] - pin_y)
                        self.net_min_max_ord[net_name]["max_y"] = pin_y
                    elif pin_y < self.net_min_max_ord[net_name]["min_y"]:
                        reward += weight * (pin_y - self.net_min_max_ord[net_name]["min_y"])
                        self.net_min_max_ord[net_name]["min_y"] = pin_y
                    start_x = self.net_min_max_ord[net_name]["min_x"]
                    end_x = self.net_min_max_ord[net_name]["max_x"]
                    start_y = self.net_min_max_ord[net_name]["min_y"]
                    end_y = self.net_min_max_ord[net_name]["max_y"]
                    delta_x = end_x - start_x
                    delta_y = end_y - start_y
                    self.rudy[start_x : end_x + 1, start_y : end_y + 1] += 1 / (delta_x + 1) + 1 / (delta_y + 1)
                else:
                    self.net_min_max_ord[net_name] = {}
                    self.net_min_max_ord[net_name]["max_x"] = pin_x
                    self.net_min_max_ord[net_name]["min_x"] = pin_x
                    self.net_min_max_ord[net_name]["max_y"] = pin_y
                    self.net_min_max_ord[net_name]["min_y"] = pin_y
                    start_x = self.net_min_max_ord[net_name]["min_x"]
                    end_x = self.net_min_max_ord[net_name]["max_x"]
                    start_y = self.net_min_max_ord[net_name]["min_y"]
                    end_y = self.net_min_max_ord[net_name]["max_y"]
                    reward += 0

        self.place_counter += 1
        done = self.place_counter == len(self.node_name_list)
        if done:
            self.state = np.zeros_like(self.state)
            next_wire_img = None
            next_pos_mask = None
        else:
            next_node_name = self.node_name_list[self.place_counter]
            next_size_x, next_size_y = self.node_orient_physical_info[next_node_name][0]["size"]

            next_wire_img = self._get_wire_img()
            next_pos_mask = self._get_pos_mask()
            self.state = self._build_state(
                canvas=canvas,
                wire_img=next_wire_img,
                pos_mask=next_pos_mask,
                place_counter=self.place_counter,
                size_x=next_size_x,
                size_y=next_size_y,
            )

        return (
            self.state,
            reward,
            done,
            {
                "raw_reward": reward,
                "wire_img": next_wire_img,
                "pos_mask": next_pos_mask,
            },
        )

    @trackit
    def _get_wire_img(self) -> np.ndarray:
        """
        生成代表线路图像的数组。

        此函数创建一个三维数组，用于表示在不同方向上节点之间的连接关系。
        它根据节点的位置和方向，以及网络的信息，来计算和更新这个三维数组。

        Returns:
            np.ndarray: 一个三维数组，表示线路的图像。
        """
        return self._get_wire_img_rust()

    def _get_wire_img_valina(self) -> np.ndarray:
        net_img = np.zeros((8, self.grid, self.grid))
        node_name = self.node_name_list[self.place_counter]
        for orient in range(8):
            for net_name in self.placedb.node2net_dict[node_name]:
                if net_name in self.net_min_max_ord:
                    pin_name = self.placedb.net_info[net_name]["nodes"][node_name]["pin_name"]
                    delta_pin_x = round(self.node_orient_physical_info[node_name][orient]["pins"][pin_name][0] / self.ratio)
                    delta_pin_y = round(self.node_orient_physical_info[node_name][orient]["pins"][pin_name][1] / self.ratio)

                    start_x = self.net_min_max_ord[net_name]["min_x"] - delta_pin_x
                    end_x = self.net_min_max_ord[net_name]["max_x"] - delta_pin_x
                    start_y = self.net_min_max_ord[net_name]["min_y"] - delta_pin_y
                    end_y = self.net_min_max_ord[net_name]["max_y"] - delta_pin_y
                    start_x = min(start_x, self.grid)
                    start_y = min(start_y, self.grid)
                    base_offset_x, base_offset_y = 0, 0
                    if end_x < 0:
                        base_offset_x = -end_x
                        end_x = 0
                    if end_y < 0:
                        base_offset_y = -end_y
                        end_y = 0
                    if not "weight" in self.placedb.net_info[net_name]:
                        weight = 1.0
                    else:
                        weight = self.placedb.net_info[net_name]["weight"]

                    for i in range(0, start_x):
                        net_img[orient, i, :] += (start_x - i) * weight
                    for j in range(0, start_y):
                        net_img[orient, :, j] += (start_y - j) * weight

                    for i in range(end_x, self.grid):
                        net_img[orient, i, :] += (i - end_x + base_offset_x) * weight
                    for j in range(end_y, self.grid):
                        net_img[orient, :, j] += (j - end_y + base_offset_y) * weight

            assert net_img[orient].min() >= 0
            max_val = net_img[orient].max()
            if max_val > 0:
                net_img[orient] /= max_val
        
        return net_img

    def _get_wire_img_rust(self) -> np.ndarray:
        import orientplace_core

        node_name = self.node_name_list[self.place_counter]
        node_net_info: dict[int, dict[str, tuple]] = {}
        for orient in range(8):
            node_net_info[orient] = {}
            for net_name in self.placedb.node2net_dict[node_name]:
                if net_name not in self.net_min_max_ord:
                    continue
                
                for pin_name in self.placedb.net_info[net_name]["nodes"][node_name]["pins"].keys():
                    delta_pin_x = round(self.node_orient_physical_info[node_name][orient]["pins"][pin_name][0] / self.ratio)
                    delta_pin_y = round(self.node_orient_physical_info[node_name][orient]["pins"][pin_name][1] / self.ratio)

                    start_x = self.net_min_max_ord[net_name]["min_x"] - delta_pin_x
                    end_x = self.net_min_max_ord[net_name]["max_x"] - delta_pin_x
                    start_y = self.net_min_max_ord[net_name]["min_y"] - delta_pin_y
                    end_y = self.net_min_max_ord[net_name]["max_y"] - delta_pin_y

                start_x = max(min(start_x, self.grid), 0)
                start_y = max(min(start_y, self.grid), 0)
                base_offset_x, base_offset_y = 0, 0
                if end_x < 0:
                    base_offset_x = -end_x
                    end_x = 0
                if end_y < 0:
                    base_offset_y = -end_y
                    end_y = 0

                weight = self.placedb.net_info[net_name].get("weight", 1.0)
                node_net_info[orient][net_name] = (
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    base_offset_x,
                    base_offset_y,
                    weight,
                )

        # res = trackit(orientplace_core.calc_wire_img)(node_net_info, self.grid)
        res = trackit(orientplace_core.calc_wire_img_parallel)(node_net_info, self.grid)
        wire_imgs = []
        for img in res:
            img = np.array(img)
            assert np.min(img) >= 0
            max_val = np.max(img)
            if max_val > 0:
                img = img / max_val
            wire_imgs.append(img)

        return np.array(wire_imgs)

    @trackit
    def _get_pos_mask(self) -> np.ndarray:
        """
        生成位置掩码矩阵
        """
        mask = np.zeros((8, self.grid, self.grid))
        current_node_name = self.node_name_list[self.place_counter]
        for orient in range(8):
            size_x, size_y = self.node_orient_physical_info[current_node_name][orient]["size"]
            size_x = math.ceil(max(1, size_x / self.ratio))
            size_y = math.ceil(max(1, size_y / self.ratio))
            for node_name in self.node_pos:
                startx = max(0, self.node_pos[node_name][0] - size_x + 1)
                starty = max(0, self.node_pos[node_name][1] - size_y + 1)
                endx = min(
                    self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1,
                    self.grid - 1,
                )
                endy = min(
                    self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1,
                    self.grid - 1,
                )
                mask[orient, startx : endx + 1, starty : endy + 1] = 1
            mask[orient, self.grid - size_x + 1 :, :] = 1
            mask[orient, :, self.grid - size_y + 1 :] = 1
        return mask

    @trackit
    def save_flyline(self, file_path: Path) -> None:
        # 参数配置
        jitter = 3  # 基础抖动幅度
        base_strength = 20  # 基础弯曲强度
        line_alpha = 0.5  # 线条透明度
        line_width = 0.8  # 线条宽度
        ref_distance = 150  # 弧度计算参考距离

        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        def generate_distinct_color(n1, n2):
            """生成高区分度的RGB颜色"""
            # 创建排序后的节点对标识
            node_pair = "".join(sorted([n1, n2]))
            # 生成哈希值并映射到色相空间
            hue = int(hashlib.md5(node_pair.encode()).hexdigest()[:5], 16) % 360
            # 转换HSV到RGB（固定饱和度80%，明度90%）
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
            return (rgb[0], rgb[1], rgb[2])  # 返回0-1范围的RGB元组

        def calculate_control_points(x1, y1, x2, y2):
            """计算带方向控制的贝塞尔曲线控制点"""
            dx = x2 - x1
            dy = y2 - y1
            distance = np.hypot(dx, dy)

            if distance == 0:
                return (x1, y1), (x2, y2)

            # 计算统一方向（始终向右凸起）
            dir_x = dy / distance  # 垂直于连线方向的单位向量
            dir_y = -dx / distance

            # 动态弯曲强度：距离越远弯曲越大
            strength = base_strength * (distance / ref_distance) ** 0.5

            # 控制点偏移计算
            cp_offset_x = dir_x * strength
            cp_offset_y = dir_y * strength

            # 设置控制点位置
            cp1 = (x1 + dx * 0.25 + cp_offset_x, y1 + dy * 0.25 + cp_offset_y)
            cp2 = (x1 + dx * 0.75 + cp_offset_x, y1 + dy * 0.75 + cp_offset_y)

            return cp1, cp2

        # 创建画布
        _, ax = plt.subplots(figsize=(20, 20))

        node2id_dict = {s: i + 1 for i, s in enumerate(self.node_name_list)}

        for node_name in self.node_pos:
            x, y, size_x, size_y, orient = self.node_pos[node_name]
            facecolor = "cyan"
            if hasattr(self.placedb, "hard_macro_info") and node_name not in self.placedb.hard_macro_info:
                facecolor = "red"

            x = round(x * self.ratio)
            y = round(y * self.ratio)
            size_x = round(size_x * self.ratio)
            size_y = round(size_y * self.ratio)
            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    size_x,  # width
                    size_y,  # height
                    linewidth=1,
                    edgecolor="k",
                    facecolor=facecolor,
                )
            )
            ax.text(
                x + size_x / 2,
                y + size_y / 2,
                f"{node2id_dict[node_name]}_{ORIENT_MAP[orient]}",
                ha="center",
                va="center",
                fontsize=8,
                color="darkblue",
            )

        # 设置画布属性
        ax.autoscale()
        ax.set_aspect("equal")
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.title("Color-Coded Network Visualization", fontsize=14)
        plt.xlabel("X Coordinate", fontsize=10)
        plt.ylabel("Y Coordinate", fontsize=10)

        # 保存图像
        plt.savefig(file_path, bbox_inches="tight", dpi=300)

        all_connections = []
        for net_name in self.placedb.net_info:
            points = []
            if os.getenv("PLACEENV_IGNORE_PORT", "0") != "1":
                for port_info in self.placedb.net_info[net_name]["ports"].values():
                    port_name = port_info["key"]
                    x, y = port_info["pin_offset"]
                    points.append((port_name, x, y))

            for node_info in self.placedb.net_info[net_name]["nodes"].values():
                node_name = node_info["key"]
                x, y = self.node_pos[node_name][:2]
                for pin_name in node_info["pins"].keys():
                    dx, dy = self.node_pin_pos[node_name][pin_name]
                    points.append((node_name, round(x * self.ratio + dx), round(y * self.ratio + dy)))

            for p1, p2 in combinations(points, 2):
                # 采样10%的边
                if random.random() > 0.01:
                    continue
                n1, x1, y1 = p1
                n2, x2, y2 = p2

                # 计算控制点
                cp1, cp2 = calculate_control_points(x1, y1, x2, y2)

                # 生成颜色
                color = generate_distinct_color(n1, n2)

                # 计算连线长度用于排序
                length = np.hypot(x2 - x1, y2 - y1)

                all_connections.append((length, [(x1, y1), cp1, cp2, (x2, y2)], color))

        # 按连线长度降序排序（先画长线）
        all_connections.sort(reverse=True, key=lambda x: x[0])

        # 绘制所有连线
        for length, points, color in all_connections:
            path = Path(
                np.array([points[0], points[1], points[2], points[3]]),
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )
            patch = patches.PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=line_width,
                alpha=line_alpha,
                capstyle="round",
            )
            ax.add_patch(patch)

        # 保存飞线图
        flyline_fig_name = file_path.with_suffix("").as_posix() + "_flyline.png"
        plt.savefig(flyline_fig_name, bbox_inches="tight", dpi=300)
        plt.close()

    @trackit
    def save_pl_file(self, file_path):
        """
        save hard macro placement resut in .pl file
        """

        count = 0
        with open(file_path, "w") as fwrite:
            for node_name in self.node_pos:
                if hasattr(self.placedb, "hard_macro_info") and node_name not in self.placedb.hard_macro_info:
                    continue
                x, y, _, _, o = self.node_pos[node_name]
                x = round((x + 1) * self.ratio)
                y = round((y + 1) * self.ratio)
                fwrite.write(f"{node_name}\t{x}\t{y}\t:\t{ORIENT_MAP[o]} /FIXED\n")
                count += 1

            print(f"placement for {count} macros has been saved to {file_path}")

    @trackit
    def calc_hpwl_and_cost(self):

        total_hpwl, total_cost = 0.0, 0.0
        for net_name in self.placedb.net_info:
            hpwl = self._calc_hpwl(net_name)
            if "weight" in self.placedb.net_info[net_name]:
                hpwl *= self.placedb.net_info[net_name]["weight"]
            total_hpwl += hpwl

            prim_cost = self._calc_prim_cost(net_name)
            if "weight" in self.placedb.net_info[net_name]:
                prim_cost *= self.placedb.net_info[net_name]["weight"]
            assert hpwl <= prim_cost + 1e-5, f"{net_name} {hpwl} {prim_cost}"
            total_cost += prim_cost
        return total_hpwl, total_cost

    def _calc_hpwl(self, net_name: str) -> float:
        max_width, max_height = self.canvas_width, self.canvas_height
        min_x, max_x = max_width * 1.1, 0.0
        min_y, max_y = max_height * 1.1, 0.0
        for node_name in self.placedb.net_info[net_name]["nodes"]:
            if node_name not in self.node_pos:
                continue
            x, y = self.node_pos[node_name][:2]
            for pin_name in self.placedb.net_info[net_name]["nodes"][node_name]["pins"].keys():
                dx, dy = self.node_pin_pos[node_name][pin_name]
                pin_x, pin_y = x * self.ratio + dx, y * self.ratio + dy

                min_x = min(pin_x, min_x)
                max_x = max(pin_x, max_x)
                min_y = min(pin_y, min_y)
                max_y = max(pin_y, max_y)

        if os.getenv("PLACEENV_IGNORE_PORT", "0") != "1":
            for port_info in self.placedb.net_info[net_name]["ports"].values():
                pin_x, pin_y = port_info["pin_offset"]

                min_x = min(pin_x, min_x)
                max_x = max(pin_x, max_x)
                min_y = min(pin_y, min_y)
                max_y = max(pin_y, max_y)

        if min_x <= max_width and min_y <= max_height:
            hpwl = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl = 0
        return hpwl

    def _calc_prim_cost(self, net_name: str) -> float:

        vertexs = []
        for node_name in self.placedb.net_info[net_name]["nodes"]:
            # 过滤掉stdcell
            if node_name not in self.node_pos:
                continue
            x, y = self.node_pos[node_name][:2]
            for pin_name in self.placedb.net_info[net_name]["nodes"][node_name]["pins"].keys():
                dx, dy = self.node_pin_pos[node_name][pin_name]
                pin_x = x * self.ratio + dx
                pin_y = y * self.ratio + dy
                node_pin_name = f"{node_name}__{pin_name}"
                vertexs.append((node_pin_name, pin_x, pin_y))

        if os.getenv("PLACEENV_IGNORE_PORT", "0") != "1":
            for port_name, port_info in self.placedb.net_info[net_name]["ports"].items():
                pin_x, pin_y = port_info["pin_offset"]
                node_pin_name = f"PORT__{port_name}"
                vertexs.append((node_pin_name, pin_x, pin_y))

        if len(vertexs) <= 1:
            return 0

        adjacent_dict: dict[str, list[tuple[int, str, str]]] = {}
        for vertex in vertexs:
            adjacent_dict[vertex[0]] = []

        for v1, v2 in combinations(vertexs, 2):
            v1_name, v1_x, v1_y = v1
            v2_name, v2_x, v2_y = v2

            weight = abs(v1_x - v2_x) + abs(v1_y - v2_y)
            adjacent_dict[v1_name].append((weight, v1_name, v2_name))
            adjacent_dict[v2_name].append((weight, v2_name, v1_name))

        start = vertexs[0][0]
        minu_tree = []
        visited = set()
        visited.add(start)
        adjacent_vertexs_edges = adjacent_dict[start]
        heapq.heapify(adjacent_vertexs_edges)
        cost = 0
        cnt = 0
        while cnt < len(vertexs) - 1:
            weight, v1, v2 = heapq.heappop(adjacent_vertexs_edges)
            if v2 not in visited:
                visited.add(v2)
                minu_tree.append((weight, v1, v2))
                cost += weight
                cnt += 1
                for next_edge in adjacent_dict[v2]:
                    if next_edge[2] not in visited:
                        heapq.heappush(adjacent_vertexs_edges, next_edge)
        return cost
