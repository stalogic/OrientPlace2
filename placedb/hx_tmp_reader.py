from pathlib import Path
from functools import cached_property
from .base_reader import DesignReader

class HxTmpReader(DesignReader):

    def __init__(self, data_root:str, design_name:str, cache_root:str, unit:int=2000,**kwargs):
        super().__init__()

        self.data_root = Path(data_root)
        self._design_name = design_name
        self.unit = unit

        try:
            cache_path = Path(cache_root)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.error_file = open(cache_path / f"{design_name}_hxtmp_reader.err", "w")
        except:
            self.error_file = None

        net_file, node_file, area_file = self._parse_hxtmp_files(self.data_root / design_name)

        self._canvas_size = self._read_canvas(area_file)
        self._node_info = self._read_node_file(node_file)
        self._net_info = self._read_net_file(net_file)
        

        if self.error_file is not None:
            self.error_file.close()

    @staticmethod
    def _parse_hxtmp_files(design_path:Path) -> tuple[Path, Path, Path]:
        assert design_path.exists(), f"{design_path} does not exist"

        net_file, node_file, area_file = None, None, None
        for file in design_path.iterdir():
            if file.is_file():
                if "net" in file.as_posix():
                    net_file = file
                elif "node" in file.as_posix():
                    node_file = file
                elif "area" in file.as_posix():
                    area_file = file

        return net_file, node_file, area_file

    @staticmethod
    def _read_canvas(area_file:Path) -> tuple[int, int]:
        assert area_file.exists(), f"{area_file} does not exist"
        width, height = 0, 0
        with open(area_file, "r") as f:
            for line in f:
                if line.startswith("DIEAREA"):
                    blocks = line.strip().split()
                    width = int(blocks[6])
                    height = int(blocks[7])
                    break

        return width, height
    
    @staticmethod
    def _read_node_file(node_file:Path) -> dict[str, dict[str, object]]:
        assert node_file.exists(), f"{node_file} does not exist"

        node_info = {}
        with open(node_file, "r") as f:
            f.readline()
            for line in f:
                blocks = line.strip().split(",")
                node_id = f"o{blocks[0]}"
                node_attribute = "FIXED" if blocks[1] == "1" else "PLACED"
                node_width = int(blocks[2]) // 2
                node_height = int(blocks[3]) // 2

                node_info[node_id] = {
                    "attributes": node_attribute,
                    "width": node_width,
                    "height": node_height,
                }

        return node_info

    @staticmethod
    def _read_net_file(net_file:Path) -> dict[str, dict[str, list[dict]]]:
        assert net_file.exists(), f"{net_file} does not exist"

        net_info = {}
        with open(net_file, "r") as f:
            f.readline()
            for line in f:
                blocks = line.strip().split(",")
                net_id = f"n{blocks[0]}"
                node_id = f"o{blocks[1]}"
                pin_direct = "OUTPUT" if blocks[2] == "1" else "INPUT"
                pin_offset = (round(float(blocks[3])/2), round(float(blocks[4])/2))
                if net_id not in net_info:
                    net_info[net_id] = {}

                if node_id not in net_info[net_id]:
                    net_info[net_id][node_id] = []
                
                net_info[net_id][node_id].append({
                    "node_id": node_id,
                    "pin_direct": pin_direct,
                    "pin_offset": pin_offset,
                })

        return net_info

    @property
    def design_name(self) -> str:
        return self._design_name
    
    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._canvas_size

    @cached_property
    def place_net_dict(self) -> dict[str, dict]:
        net_dict = {}
        for net_name in self._net_info:
            if net_name not in net_dict:
                net_dict[net_name] = {
                    "id": len(net_dict),
                    "key": net_name,
                    "nodes": [],
                    "ports": [],
                    "weight": 1.0,
                }
            
            for node_name in self._net_info[net_name]:
                for pin_info in self._net_info[net_name][node_name]:
                    pin_direct = pin_info["pin_direct"]
                    pin_offset = pin_info["pin_offset"]
                    node_info = self._node_info[node_name]
                    width = node_info["width"]
                    height = node_info["height"]
                    pin_offset_x = round(pin_offset[0] + width / 2)
                    pin_offset_y = round(pin_offset[1] + height / 2)
                    assert pin_offset_x >= 0 and pin_offset_y >= 0
                    pin_name = f"{pin_direct[0]}_{pin_offset_x}_{pin_offset_y}"

                    net_dict[net_name]["nodes"].append((node_name, pin_name))
            
        return net_dict
    
    @cached_property
    def place_node_dict(self) -> dict:
        node_dict = {}
        for node_name in self._node_info:
            node_dict[node_name] = {
                "coordinate" : None,
                "orientation" : None,
                "attributes" : self._node_info[node_name]["attributes"],
                "node_type" : node_name.upper(),
            }
        return node_dict
    
    @cached_property
    def place_port_dict(self) -> dict:
        return {}
    
    @cached_property
    def node_phisical_info(self) -> dict:
        node_phisical_info = {}
        for net_name in self._net_info:
            for node_name in self._net_info[net_name]:
                
                node_type = node_name.upper()

                if node_type not in node_phisical_info:
                    node_phisical_info[node_type] = {}
                    node_phisical_info[node_type]["pins"] = {}

                if "size" not in node_phisical_info[node_type]:
                    assert node_name in self._node_info
                    node_info = self._node_info[node_name]
                    width = node_info["width"]
                    height = node_info["height"]
                    node_phisical_info[node_type]["size"] = (width, height)

                for pin_info in self._net_info[net_name][node_name]:
                    pin_direct = pin_info["pin_direct"]
                    pin_offset = pin_info["pin_offset"]
                    pin_offset_x = round(pin_offset[0] + node_phisical_info[node_type]["size"][0] / 2)
                    pin_offset_y = round(pin_offset[1] + node_phisical_info[node_type]["size"][1] / 2)
                    assert pin_offset_x >= 0 and pin_offset_y >= 0

                    pin_name = f"{pin_direct[0]}_{pin_offset_x}_{pin_offset_y}"
                    assert pin_name not in node_phisical_info[node_type]["pins"]
                    node_phisical_info[node_type]["pins"][pin_name] = {
                        "pin_offset": (pin_offset_x, pin_offset_y),
                        "direction": pin_direct,
                    }

        return node_phisical_info


if __name__ == "__main__":
    data_root = "/home/jiangmingming/data/OrientPlace/benchmark"
    design_name = "blackparrot"
    cache_root = "/home/jiangmingming/data/OrientPlace/cache"

    reader = HxTmpReader(data_root, design_name, cache_root)
    print(reader.canvas_size)
    print(reader.design_name)
    print(len(reader._node_info))
    fn = lambda x: sum(len(pin) for pin in x.values())
    print(sum([fn(net) for net in reader._net_info.values()]))

    index = 0
    for node_type in reader.node_phisical_info:
        print(node_type, reader.node_phisical_info[node_type])
        
        index += 1
        if index > 10:
            break


    index = 0
    for net_name in reader.place_net_dict:
        print(net_name, reader.place_net_dict[net_name])
        
        index += 1
        if index > 10:
            break

    index = 0
    for node_name in reader.place_node_dict:
        if reader.place_node_dict[node_name]['attributes'] == 'PLACED':
            continue
        print(node_name, reader.place_node_dict[node_name])
        
        index += 1
        if index > 10:
            break