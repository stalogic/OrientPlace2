from itertools import combinations
from .base_placedb import BasePlaceDB
from .base_reader import DesignReader
from functools import cached_property
import os
import math

class PlaceDB(BasePlaceDB):

    def __init__(self, reader: DesignReader):
        self._reader = reader

    @property
    def design_name(self) -> str:
        return self._reader.design_name
    
    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._reader.canvas_size

    @cached_property
    def macro_info(self) -> dict[str, dict[str, object]]:
        macro_info = {}
        macro_info.update(self.hard_macro_info)
        macro_info.update(self.soft_macro_info)
        return macro_info
    
    @cached_property
    def hard_macro_info(self) -> dict[str, dict[str, object]]:
        hard_macro_info = {}
        for node_name in self._reader.place_node_dict:
            if self._reader.place_node_dict[node_name]["attributes"] == "FIXED":
                
                node_type = self._reader.place_node_dict[node_name]["node_type"]
                width, height = self._reader.node_phisical_info[node_type]["size"]
                hard_macro_info[node_name] = {
                    "node_type": node_type,
                    "width": width,
                    "height": height,
                }
        return hard_macro_info
    
    @cached_property
    def soft_macro_info(self) -> dict[str, dict[str, object]]:
        soft_macro_info = {}
        for node_name in self._reader.place_node_dict:
            if self._reader.place_node_dict[node_name]["attributes"] == "VIRTUAL":
                
                node_type = self._reader.place_node_dict[node_name]["node_type"]
                width, height = self._reader.node_phisical_info[node_type]["size"]
                soft_macro_info[node_name] = {
                    "node_type": node_type,
                    "width": width,
                    "height": height,
                }
        return soft_macro_info
    
    @cached_property
    def cell_info(self) -> dict[str, dict[str, object]]:
        cell_info = {}
        for node_name in self._reader.place_node_dict:
            if self._reader.place_node_dict[node_name]["attributes"] == "PLACED":
                
                node_type = self._reader.place_node_dict[node_name]["node_type"]
                width, height = self._reader.node_phisical_info[node_type]["size"]
                cell_info[node_name] = {
                    "node_type": node_type,
                    "width": width,
                    "height": height,
                }
        return cell_info
        
    
    @cached_property
    def node_info(self) -> dict[str, dict[str, object]]:
        node_info = {}
        for node_name in self._reader.place_node_dict:                
            node_type = self._reader.place_node_dict[node_name]["node_type"]
            width, height = self._reader.node_phisical_info[node_type]["size"]
            node_info[node_name] = {
                "node_type": node_type,
                "width": width,
                "height": height,
            }
        return node_info
    
    @cached_property
    def net_info(self) -> dict[str, dict[str, object]]:
        net_info :dict[str, dict[str, object]] = {}
        for net_name, net_dict in self._reader.place_net_dict.items():

            net_info[net_name] = {
                "id": net_dict["id"],
                "key": net_name,
                "source": {},
                "nodes": {},
                "ports": {}
            }

            for node_name, pin_name in net_dict["nodes"]:
                node_type = self._reader.place_node_dict[node_name]["node_type"]
                node_info = self._reader.node_phisical_info[node_type]
                pin_offset = node_info["pins"][pin_name]["pin_offset"]
                pin_direct = node_info["pins"][pin_name]["direction"]

                if node_name not in net_info[net_name]["nodes"]:
                    net_info[net_name]["nodes"][node_name] = {"key": node_name, "node_type": node_type, "pins": {}}
                net_info[net_name]["nodes"][node_name]["pins"][pin_name] = {"key": pin_name, "pin_offset": pin_offset}

                if pin_direct == "OUTPUT":
                    if os.getenv("NET_SOURCE_CHECK", "1") != "0":
                        assert len(net_info[net_name]["source"]) == 0, "net has more then one input pin, set Environment Variable `NET_SOURCE_CHECK=0` to disable this check"
                    net_info[net_name]["source"]["node_name"] = node_name
                    net_info[net_name]["source"]["node_type"] = node_type
                    net_info[net_name]["source"]["pin_name"] = pin_name

            for port_name in net_dict["ports"]:
                port_info = self._reader.place_port_dict[port_name]
                coordinate = port_info["coordinate"]
                port_direct = port_info["direction"]
                net_info[net_name]["ports"][port_name] = {
                    "key": port_name,
                    "pin_name": port_name,
                    "node_type": "PIN",
                    "pin_offset": coordinate,
                }

                if port_direct == "INPUT":
                    if os.getenv("NET_SOURCE_CHECK", "1") != "0":
                        assert len(net_info[net_name]["source"]) == 0, "net has more then one input pin, set Environment Variable `NET_SOURCE_CHECK=0` to disable this check"
                    net_info[net_name]["source"]["node_name"] = port_name
                    net_info[net_name]["source"]["node_type"] = "PIN"
                    net_info[net_name]["source"]["pin_name"] = port_name

        return net_info

    @cached_property
    def port_info(self) -> dict[str, dict[str, object]]:
        port_info = {}
        for port_name, port_dict in self._reader.place_port_dict.items():
            coordinate = port_dict["coordinate"]
            direction = port_dict["direction"]
            port_info[port_name] = {
                "coordinate": coordinate,
                "direction": direction,
                "orientation": None,
            }

        return port_info
    
    @cached_property
    def node2net_dict(self) -> dict[str, set[str]]:
        node2net_dict = {}
        for net_name, net_info in self._reader.place_net_dict.items():
            for node_name, _ in net_info["nodes"]:
                if node_name not in node2net_dict:
                    node2net_dict[node_name] = set()
                node2net_dict[node_name].add(net_name)
        return node2net_dict

    @cached_property
    def port2net_dict(self) -> dict[str, set[str]]:
        port2net_dict :dict[str, set[str]] = {}
        for net_name, net_info in self._reader.place_net_dict.items():
            for port_name in net_info["ports"]:
                if port_name not in port2net_dict:
                    port2net_dict[port_name] = set()
                port2net_dict[port_name].add(net_name)
        return port2net_dict
        
    @cached_property
    def macro_place_queue(self) -> list[str]:
        macro_adjancy = {}
        for macro1, macro2 in combinations(self.macro_info, 2):
            macro1_nets = set(self.node2net_dict[macro1])
            macro2_nets = set(self.node2net_dict[macro2])
            common_nets = len(macro1_nets & macro2_nets)

            if macro1 not in macro_adjancy:
                macro_adjancy[macro1] = {}
            if macro2 not in macro_adjancy:
                macro_adjancy[macro2] = {}

            if common_nets > 0:
                macro_adjancy[macro1][macro2] = common_nets
                macro_adjancy[macro2][macro1] = common_nets
        
        target_set = set()
        residual_set = set(self.macro_info)
        macro_queue = []

        while len(residual_set) > 0:
            max_macro, max_weight = None, None
            for macro in residual_set:
                assert macro_adjancy[macro].get(macro, 0) == 0
                residual_weight = sum([macro_adjancy[macro].get(other, 0) for other in residual_set])
                target_weight = sum([macro_adjancy[macro].get(other, 0) for other in target_set])

                weight = target_weight * residual_weight
                area = self.macro_info[macro]["width"] * self.macro_info[macro]["height"]
                weight_level = round(math.log(weight + 1))
                area_level = round(math.log(area + 1))
                weight = (weight_level, area_level, area, weight)

                if max_macro is None or weight > max_weight:
                    max_macro, max_weight = macro, weight
            assert max_macro is not None
            macro_queue.append(max_macro)
            target_set.add(max_macro)
            residual_set.remove(max_macro)

        assert len(macro_queue) == len(self.macro_info)
        return macro_queue