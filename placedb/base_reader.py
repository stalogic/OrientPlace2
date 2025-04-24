from abc import ABC, abstractmethod


class DesignReader(ABC):

    @property
    @abstractmethod
    def design_name(self) -> str:
        """
        Design name
        """
        ...

    @property
    @abstractmethod
    def canvas_size(self) -> tuple[int, int]:
        """
        Chip canvas size
        """
        ...

    @property
    @abstractmethod
    def place_net_dict(self) -> dict:
        """
        Net information

        Returns
        dict: A dictionary containing the net information of the chip, including net name, net nodes, and net weight.
        ```
        # Net weight is used to calculate the net weight in the chip
        {
            net_name: {
                "id": net_id,
                "key": net_name,
                "weight": net_weight,
                "nodes": [(node_name, pin_name), ...],
                "ports": [port_name1, port_name2, ...],
            }
        }
        """
        pass

    @property
    @abstractmethod
    def place_node_dict(self) -> dict:
        """
        Plcaed node information

        Returns
        dict: A dictionary containing the node information of the chip, including node name, node type, and node coordinate.
        ```
        # 
        {
            node_name: {
                "coordinate": (x, y),
                "orientation": "N" | "W" | "S" | "E" | "FS" | "FE" | "FN" | "FW",
                "attributes": "FIXED" | "PLACED" | "VIRTUAL",
                "node_type": str,
            }
        }
        """
        pass

    @property
    @abstractmethod
    def place_port_dict(self) -> dict:
        """
        Chip port information

        Returns
        dict: A dictionary containing the port information of the chip, including port name, port type, and port coordinate.
        ```
        # port coordinate is relative to the chip left bottom corner
        # port direction is reversed when used in a net, 
        # e.g. 
        #       An INPUT port in net means the port outputs signial to other node
        #       An OUTPUT port in net means the port inputs signal from other node
        {
            port_name: {
                "coordinate": (x, y),
                "direction": "INPUT" | "OUTPUT", 
            }
        }
        """
        pass

    @property
    @abstractmethod
    def node_phisical_info(self) -> dict:
        """
        Node type physical information

        Returns
        dict: A dictionary containing the physical information of the node type, including node size and pin information.
        ```
        # Node size, including width and height, shuold multiply by unit_size
        # Pin coordinate offset is relative to the node left bottom corner
        {
            node_type: {
                "size": (width, height), 
                "pins": {
                    pin_name: {
                        "pin_offset": (x, y), 
                        "direction": "INPUT" | "OUTPUT" | "INOUT",
                    }
                }
            }
        }
        """
        pass