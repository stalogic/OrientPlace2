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
        # 记录每个net的范围的最新信息, key1为net_name，key2为`min_x`, `max_x`, `min_y`, `max_y`
        self.net_min_max_ord: dict[str, dict[str, int]] = {}
        # 记录每个net的范围的次新信息, key1为net_name，key2为`min_x`, `max_x`, `min_y`, `max_y`
        self.last_net_min_max_ord: dict[str, dict[str, int]] = {}
        # 记录每个net的支持node，pin。key1为net_name，key2为`min_x`, `max_x`, `min_y`, `max_y`
        # 元组内为node_name, pin_name 或者 "PORT", port_name
        self.net_support_node_pin: dict[str, dict[str, tuple[str, str]]] = {}
        # 记录每个node的位置信息，即左下角的x坐标，y坐标，宽度w，高度h，方向o
        self.node_pos: dict[str, tuple[int, int, int, int, int]] = {}
        # 记录每个node的每个pin的相对坐标信息，key1为`PORT`时表示记录芯片对外IO端口
        self.node_pin_pos: dict[str, dict[str, tuple[float, float]]] = {}
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
        ```
        {
            node_name: {
                orient: {
                    'size': (width, height),
                    'pins': {
                        pin_name: (offset_x, offset_y),
                    }
                }
            }
        }
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
        self.rudy = np.zeros((self.grid, self.grid))
        self.node_pos.clear()
        self.net_min_max_ord.clear()
        self.last_net_min_max_ord.clear()
        self.net_support_node_pin.clear()
        self.node_pin_pos.clear()
        
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
                            self.last_net_min_max_ord[net_name]["max_x"] = self.net_min_max_ord[net_name]["max_x"]
                            self.net_min_max_ord[net_name]["max_x"] = pin_x
                            self.net_support_node_pin[net_name]["max_x"] = ("PORT", port_name)
                        elif pin_x < self.net_min_max_ord[net_name]["min_x"]:
                            self.last_net_min_max_ord[net_name]["min_x"] = self.net_min_max_ord[net_name]["min_x"]
                            self.net_min_max_ord[net_name]["max_y"] = pin_y
                            self.net_support_node_pin[net_name]["max_y"] = ("PORT", port_name)
                        if pin_y > self.net_min_max_ord[net_name]["max_y"]:
                            self.last_net_min_max_ord[net_name]["max_y"] = self.net_min_max_ord[net_name]["max_y"]
                            self.net_min_max_ord[net_name]["max_y"] = pin_y
                            self.net_support_node_pin[net_name]["max_y"] = ("PORT", port_name)
                        elif pin_y < self.net_min_max_ord[net_name]["min_y"]:
                            self.last_net_min_max_ord[net_name]["min_y"] = self.net_min_max_ord[net_name]["min_y"]
                            self.net_min_max_ord[net_name]["min_y"] = pin_y
                            self.net_support_node_pin[net_name]["min_y"] = ("PORT", port_name)
                    else:
                        self.net_min_max_ord[net_name] = {}
                        self.last_net_min_max_ord[net_name] = {}
                        self.net_support_node_pin[net_name] = {}
                        self.net_min_max_ord[net_name]["max_x"] = pin_x
                        self.net_min_max_ord[net_name]["min_x"] = pin_x
                        self.net_min_max_ord[net_name]["max_y"] = pin_y
                        self.net_min_max_ord[net_name]["min_y"] = pin_y
                        self.net_support_node_pin[net_name]["max_x"] = ("PORT", port_name)
                        self.net_support_node_pin[net_name]["min_x"] = ("PORT", port_name)
                        self.net_support_node_pin[net_name]["max_y"] = ("PORT", port_name)
                        self.net_support_node_pin[net_name]["min_y"] = ("PORT", port_name)

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
                        self.last_net_min_max_ord[net_name]["max_x"] = self.net_min_max_ord[net_name]["max_x"]
                        self.net_min_max_ord[net_name]["max_x"] = pin_x
                        self.net_support_node_pin[net_name]["max_x"] = (node_name, pin_name)
                    elif pin_x < self.net_min_max_ord[net_name]["min_x"]:
                        reward += weight * (pin_x - self.net_min_max_ord[net_name]["min_x"])
                        self.last_net_min_max_ord[net_name]["min_x"] = self.net_min_max_ord[net_name]["min_x"]
                        self.net_min_max_ord[net_name]["min_x"] = pin_x
                        self.net_support_node_pin[net_name]["min_x"] = (node_name, pin_name)
                    if pin_y > self.net_min_max_ord[net_name]["max_y"]:
                        reward += weight * (self.net_min_max_ord[net_name]["max_y"] - pin_y)
                        self.last_net_min_max_ord[net_name]["max_y"] = self.net_min_max_ord[net_name]["max_y"]
                        self.net_min_max_ord[net_name]["max_y"] = pin_y
                        self.net_support_node_pin[net_name]["max_y"] = (node_name, pin_name)
                    elif pin_y < self.net_min_max_ord[net_name]["min_y"]:
                        reward += weight * (pin_y - self.net_min_max_ord[net_name]["min_y"])
                        self.last_net_min_max_ord[net_name]["min_y"] = self.net_min_max_ord[net_name]["min_y"]
                        self.net_min_max_ord[net_name]["min_y"] = pin_y
                        self.net_support_node_pin[net_name]["min_y"] = (node_name, pin_name)
                else:
                    self.net_min_max_ord[net_name] = {}
                    self.last_net_min_max_ord[net_name] = {}
                    self.net_support_node_pin[net_name] = {}
                    self.net_min_max_ord[net_name]["max_x"] = pin_x
                    self.net_min_max_ord[net_name]["min_x"] = pin_x
                    self.net_min_max_ord[net_name]["max_y"] = pin_y
                    self.net_min_max_ord[net_name]["min_y"] = pin_y

                    self.net_support_node_pin[net_name]["max_x"] = (node_name, pin_name)
                    self.net_support_node_pin[net_name]["min_x"] = (node_name, pin_name)
                    self.net_support_node_pin[net_name]["max_y"] = (node_name, pin_name)
                    self.net_support_node_pin[net_name]["min_y"] = (node_name, pin_name)
                    # 当net里只有一个点时，不会引起hpwl增加
                    # reward += 0
                
                # 计算rudy值，估计congestion
                start_x = self.net_min_max_ord[net_name]["min_x"]
                end_x = self.net_min_max_ord[net_name]["max_x"]
                start_y = self.net_min_max_ord[net_name]["min_y"]
                end_y = self.net_min_max_ord[net_name]["max_y"]
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                if delta_x > 0 or delta_y > 0:
                    self.rudy[start_x : end_x + 1, start_y : end_y + 1] += 1 / (delta_x + 1) + 1 / (delta_y + 1)

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

        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        colors = ["green", "blue", "yellow", "red"]
        node_orient_metric = self._calc_node_orient_metric()

        # 创建画布
        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40, 20))

        # 画node
        node2id_dict = {s: i + 1 for i, s in enumerate(self.node_name_list)}
        for node_name in self.node_pos:
            x, y, _, _, orient = self.node_pos[node_name]
            facecolor = "cyan"
            if hasattr(self.placedb, "hard_macro_info") and node_name not in self.placedb.hard_macro_info:
                facecolor = "yellow"

            x = round(x * self.ratio)
            y = round(y * self.ratio)
            size_x, size_y = self.node_orient_physical_info[node_name][orient]["size"]
            ax1.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    size_x,  # width
                    size_y,  # height
                    linewidth=1,
                    edgecolor="k",
                    facecolor=facecolor,
                )
            )
            ax1.text(
                x + size_x / 2,
                y + size_y / 2,
                f"{node2id_dict[node_name]}_{ORIENT_MAP[orient]}",
                ha="center",
                va="center",
                fontsize=8,
                color="darkblue",
            )

            facecolor = "white"
            content = ""
            if node_name in node_orient_metric:
                sorted_index, grid_hpwl, min_hpwl, max_hpwl = node_orient_metric[node_name]
                facecolor = colors[sorted_index]
                ratio1 = int((grid_hpwl/min_hpwl-1) * 1000)
                ratio2 = int((max_hpwl/min_hpwl-1) * 1000)
                content = f"{ratio1}_{ratio2}"

            ax2.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    size_x,  # width
                    size_y,  # height
                    linewidth=1,
                    edgecolor="k",
                    facecolor=facecolor,
                )
            )
            ax2.text(
                x + size_x / 2,
                y + size_y / 2,
                content,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        # 画pin
        pin_points = []
        for node_name in self.node_pin_pos:
            if node_name == "PORT":
                for port_name in self.node_pin_pos[node_name]:
                    pin_x, pin_y = self.node_pin_pos[node_name][port_name]
                    pin_points.append((pin_x, pin_y))
            else:
                x, y = self.node_pos[node_name][:2]
                for pin_name in self.node_pin_pos[node_name]:
                    dx, dy = self.node_pin_pos[node_name][pin_name]
                    pin_x = round(x * self.ratio + dx)
                    pin_y = round(y * self.ratio + dy)
                    pin_points.append((pin_x, pin_y))
        ax1.plot(*zip(*pin_points), "o", markersize=1, color="red")

        # 设置画布属性
        ax1.autoscale()
        ax1.set_aspect("equal")
        ax1.grid(True, linestyle=":", alpha=0.3)
        ax1.set_title("Color-Coded Node Placement Visualization", fontsize=14)
        ax1.set_xlabel("X Coordinate", fontsize=10)
        ax1.set_ylabel("Y Coordinate", fontsize=10)

        ax2.autoscale()
        ax2.set_aspect("equal")
        ax2.grid(True, linestyle=":", alpha=0.3)
        ax2.set_title("Color-Coded Node Orientation Visualization", fontsize=14)
        ax2.set_xlabel("X Coordinate", fontsize=10)
        ax2.set_ylabel("Y Coordinate", fontsize=10)

        # 保存图像
        plt.savefig(file_path, bbox_inches="tight", dpi=100)
        plt.close()

    @trackit
    def _calc_node_orient_metric(self):
        node_orient_metric = {}
        for net_name in self.net_support_node_pin:
            for node_name, pin_name in self.net_support_node_pin[net_name].values():
                if node_name not in self.placedb.macro_info:
                    continue
                if node_name not in node_orient_metric:
                    node_orient_metric[node_name] = {"pins": set()}
                node_orient_metric[node_name]["pins"].add(pin_name)
        macro_count = len(self.placedb.macro_info)
        support_node_count = len(node_orient_metric)
        support_pin_count = sum([len(v["pins"]) for v in node_orient_metric.values()])
        net_count = len(self.net_support_node_pin)
        print(f"Macro count: {macro_count}, Support node count: {support_node_count} Support pin count: {support_pin_count} Net num: {net_count}")

        node_metric_dict = {}
        for node_name in node_orient_metric:
            # 计算当前node在4种orient下的grid_hpwl
            node_grid_hpwl = []
            node_x, node_y, _, _, current_orient = self.node_pos[node_name]
            for index in range(4):
                orient = (current_orient + index * 2) % 8
                grid_hpwl = 0
                for net_name in self.placedb.node2net_dict[node_name]:
                    current_max_x = self.net_min_max_ord[net_name]["max_x"]
                    current_min_x = self.net_min_max_ord[net_name]["min_x"]
                    current_max_y = self.net_min_max_ord[net_name]["max_y"]
                    current_min_y = self.net_min_max_ord[net_name]["min_y"]
                    last_max_x = self.last_net_min_max_ord[net_name].get("max_x", -1)
                    last_min_x = self.last_net_min_max_ord[net_name].get("min_x", self.grid+1)
                    last_max_y = self.last_net_min_max_ord[net_name].get("max_y", -1)
                    last_min_y = self.last_net_min_max_ord[net_name].get("min_y", self.grid+1)

                    max_x, min_x, max_y, min_y = [], [], [], []

                    for pin_name in self.placedb.net_info[net_name]["nodes"][node_name]["pins"]:
                        pin_offset_x, pin_offset_y = self.node_orient_physical_info[node_name][orient]["pins"][pin_name]
                        pin_x = round((node_x * self.ratio + pin_offset_x) / self.ratio)
                        pin_y = round((node_y * self.ratio + pin_offset_y) / self.ratio)
                        
                        if current_min_x < pin_x < current_max_x:
                            # 减小hpwl，需要先确认该pin support了 max_x 或 min_x
                            if self.net_support_node_pin[net_name]["max_x"] == (node_name, pin_name):
                                max_x.append(max(pin_x, last_max_x))
                            else:
                                max_x.append(current_max_x)

                            if self.net_support_node_pin[net_name]["min_x"] == (node_name, pin_name):
                                min_x.append(min(pin_x, last_min_x))
                            else:
                                min_x.append(current_min_x)
                        else:
                            # 扩大hpwl
                            max_x.append(max(pin_x, current_max_x))
                            min_x.append(min(pin_x, current_min_x))

                        if current_min_y < pin_y < current_max_y:
                            # 减小hpwl，需要先确认该pin support了 max_y 或 min_y
                            if self.net_support_node_pin[net_name]["max_y"] == (node_name, pin_name):
                                max_y.append(max(pin_y, last_max_y))
                            else:
                                max_y.append(current_max_y)

                            if self.net_support_node_pin[net_name]["min_y"] == (node_name, pin_name):
                                min_y.append(min(pin_y, last_min_y))
                            else:
                                min_y.append(current_min_y)
                        else:
                            # 扩大hpwl
                            max_y.append(max(pin_y, current_max_y))
                            min_y.append(min(pin_y, current_min_y))

                    grid_hpwl += max(max_x) - min(min_x) + max(max_y) - min(min_y)
                node_grid_hpwl.append(grid_hpwl)

            sorted_index = sorted(node_grid_hpwl).index(node_grid_hpwl[0])
            node_metric_dict[node_name] = (sorted_index, node_grid_hpwl[0], min(node_grid_hpwl), max(node_grid_hpwl))
        
        return node_metric_dict

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
