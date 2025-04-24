import math
import igraph as ig
from loguru import logger

from .base_reader import DesignReader

class SoftMacroReader(DesignReader):

    def __init__(self, reader: DesignReader, subgraph_list: list[ig.Graph], gamma=1.1):
        super().__init__()
        self._design_name = reader.design_name
        self._canvas_size = reader.canvas_size
        self._place_net_dict = {}
        self._place_node_dict = {}
        self._place_port_dict = {}
        self._node_phisical_info = {}

        node_type_set = set()
        for node_name, node_info in reader.place_node_dict.items():
            if node_info["attributes"] == "FIXED":
                self._place_node_dict[node_name] = node_info
                node_type_set.add(node_info["node_type"])

        for node_type in node_type_set:
            self._node_phisical_info[node_type] = reader.node_phisical_info[node_type]

        self._place_port_dict.update(reader.place_port_dict)

        
        # 处理node相关信息
        node2soft_macro = {}
        soft_macro_list = []
        for i, subgraph in enumerate(subgraph_list):
            area = sum(subgraph.vs['area'])
            macro_name = f"virtual/soft_macro_{i}"
            macro_type = f"SOFT_MACRO_{i}"
            macro = {
                "coordinate": (0, 0),
                "orientation": "N",
                "attributes": "VIRTUAL",
                "node_type": macro_type,
            }
            self._place_node_dict[macro_name] = macro
            soft_macro_list.append(macro_name)
            for node in subgraph.vs["name"]:
                node2soft_macro[node] = macro_name

            width = round(math.sqrt(area) * gamma)
            self._node_phisical_info[macro_type] = {
                "size": [width, width],
                "pins": {
                    "virtual_pin": {
                        "pin_offset": (width/2, width/2),
                        "direction": "INOUT"
                    }
                },
            }

        # 处理net相关信息
        valid_macro = set()
        for net_name in reader.place_net_dict:
            node_set = set()
            nodes = []
            for node_name, pin_name in reader.place_net_dict[net_name]["nodes"]:
                if node_name in node2soft_macro:
                    # net中，属于同一个soft_macro的node，只保留一次，并使用soft_macro代替
                    soft_macro = node2soft_macro[node_name]
                    if soft_macro not in node_set:
                        nodes.append((soft_macro, "virtual_pin"))
                        node_set.add(soft_macro)
                elif node_name in self._place_node_dict:
                    nodes.append((node_name, pin_name))
                else:
                    # discard node
                    pass
            # 仅保留hard macro或soft macro总数大于2的net
            if len(nodes) + len(reader.place_net_dict[net_name]["ports"]) >= 2:
                valid_macro.update([n[0] for n in nodes])
                self._place_net_dict[net_name] = {
                    "id": len(self._place_net_dict),
                    "key": net_name,
                    "weight": reader.place_net_dict[net_name].get("weight", 1),
                    "nodes": nodes,
                    "ports": reader.place_net_dict[net_name]["ports"],
                }

        # 移除没有合法net的soft macro
        node_list = list(self._place_node_dict)
        for macro in node_list:
            if macro not in valid_macro:
                del self._place_node_dict[macro]
                if macro not in soft_macro_list:
                    logger.warning(f"Remove hard macro {macro} for not in any net.")

    @property
    def design_name(self) -> str:
        return f"{self._design_name}^soft_macro"
    
    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._canvas_size

    @property
    def place_net_dict(self) -> dict:
        return self._place_net_dict
    
    @property
    def place_node_dict(self) -> dict:
        return self._place_node_dict
    
    @property
    def place_port_dict(self) -> dict:
        return self._place_port_dict
    
    @property
    def node_phisical_info(self) -> dict:
        return self._node_phisical_info
