import binascii
import os
import random

import numpy as np
import torch


def is_gzip_file(file_path):
    with open(file_path, "rb") as file:
        header = file.read(2)
    hex_header = binascii.hexlify(header).decode("utf-8")
    if hex_header == "1f8b":
        return True
    else:
        return False


def my_range(start, end):
    if start == end:
        return [start]
    if start != end:
        return range(start, end)


def instance_direction_rect(
    line,
):  # used when we only need bounding box (rect) of the cell.
    if "N" in line or "S" in line:
        m_direction = (1, 0, 0, 1)
    elif "W" in line or "E" in line:
        m_direction = (0, 1, 1, 0)
    else:
        raise ValueError("read_macro_direction_wrong")
    return m_direction


def save_numpy(root_path, dir_name, save_name, data):
    save_path = os.path.join(root_path, dir_name, save_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, data)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def orient_pin_transform(orient: int, size: tuple[float, float], pin: tuple[float, float]):
    assert isinstance(orient, int), f"orientation must be int, but got {orient}"
    width, height = size
    x, y = pin
    if orient == 0:
        return x, y
    elif orient == 1:
        return height - y, x
    elif orient == 2:
        return width - x, height - y
    elif orient == 3:
        return y, width - x
    elif orient == 4:
        return x, height - y
    elif orient == 5:
        return y, x
    elif orient == 6:
        return width - x, y
    elif orient == 7:
        return height - y, width - x
    else:
        raise ValueError(f"orientation must be in range [0, 7], but got {orient}")


def orient_size_transform(orient: int, size: tuple[float, float]) -> tuple[float, float]:
    assert isinstance(orient, int) and 0 <= orient < 8, f"orientation must be in range [0, 7], but got {orient}"
    return size if orient % 2 == 0 else (size[1], size[0])
