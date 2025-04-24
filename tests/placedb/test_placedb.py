# tests/test_general_placedb.py
import unittest
from unittest.mock import MagicMock
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from placedb import PlaceDB

class TestPlaceDB(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        # 模拟 DesignReader 类
        self.mock_reader = MagicMock()
        self.mock_reader.design_name = "test_design"
        self.mock_reader.place_node_dict = {
            "node1": {"attributes": "FIXED", "node_type": "type1"},
            "node2": {"attributes": "VIRTUAL", "node_type": "type2"},
            "node3": {"attributes": "PLACED", "node_type": "type3"},
        }
        self.mock_reader.node_phisical_info = {
            "type1": {"size": (100, 200), "pins": {"pin1": {"pin_offset": (10, 20), "direction": "INPUT"}}},
            "type2": {"size": (150, 250), "pins": {"pin2": {"pin_offset": (15, 25), "direction": "OUTPUT"}}},
            "type3": {"size": (200, 300), "pins": {"pin3": {"pin_offset": (20, 30), "direction": "INPUT"}}},
        }
        self.mock_reader.place_net_dict = {
            "net1": {
                "id": 1,
                "nodes": [("node1", "pin1"), ("node3", "pin3")],
                "ports": ["port1"],
            },
        }
        self.mock_reader.place_port_dict = {
            "port1": {"coordinate": (30, 40), "direction": "INPUT"},
        }

        # 创建 GeneralPlaceDB 实例
        self.placedb = PlaceDB(self.mock_reader)

    def test_design_name(self):
        self.assertEqual(self.placedb.design_name, "test_design")

    def test_cache_tag(self):
        self.assertEqual(self.placedb.cache_tag, "test_design_H1_S1_C1_P1_N1")

    def test_macro_info(self):
        expected_macro_info = {
            "node1": {"node_type": "type1", "width": 100, "height": 200},
            "node2": {"node_type": "type2", "width": 150, "height": 250},
        }
        self.assertEqual(self.placedb.macro_info, expected_macro_info)

    def test_hard_macro_info(self):
        expected_hard_macro_info = {
            "node1": {"node_type": "type1", "width": 100, "height": 200},
        }
        self.assertEqual(self.placedb.hard_macro_info, expected_hard_macro_info)

    def test_soft_macro_info(self):
        expected_soft_macro_info = {
            "node2": {"node_type": "type2", "width": 150, "height": 250},
        }
        self.assertEqual(self.placedb.soft_macro_info, expected_soft_macro_info)

    def test_cell_info(self):
        expected_cell_info = {
            "node3": {"node_type": "type3", "width": 200, "height": 300},
        }
        self.assertEqual(self.placedb.cell_info, expected_cell_info)

    def test_node_info(self):
        expected_node_info = {
            "node1": {"node_type": "type1", "width": 100, "height": 200},
            "node2": {"node_type": "type2", "width": 150, "height": 250},
            "node3": {"node_type": "type3", "width": 200, "height": 300},
        }
        self.assertEqual(self.placedb.node_info, expected_node_info)

    def test_net_info(self):
        expected_net_info = {
            "net1": {
                "id": 1,
                "key": "net1",
                "source": {"node_name": "port1", "node_type": "PIN", "pin_name": "port1"},
                "nodes": {
                    "node1": {"key": "node1", "node_type": "type1", "pins": {"pin1": {"key": "pin1", "pin_offset": (10, 20)}}},
                    "node3": {"key": "node3", "node_type": "type3", "pins": {"pin3": {"key": "pin3", "pin_offset": (20, 30)}}},
                },
                "ports": {
                    "port1": {"key": "port1", "pin_name": "port1", "node_type": "PIN", "pin_offset": (30, 40)},
                },
            },
        }
        self.assertEqual(self.placedb.net_info, expected_net_info)

    def test_port_info(self):
        expected_port_info = {
            "port1": {"coordinate": (30, 40), "direction": "INPUT", "orientation": None},
        }
        self.assertEqual(self.placedb.port_info, expected_port_info)

    def test_node2net_dict(self):
        expected_node2net_dict = {
            "node1": {"net1"},
            "node3": {"net1"},
        }
        self.assertEqual(self.placedb.node2net_dict, expected_node2net_dict)

    def test_port2net_dict(self):
        expected_port2net_dict = {
            "port1": {"net1"},
        }
        self.assertEqual(self.placedb.port2net_dict, expected_port2net_dict)

    def test_macro_place_queue(self):
        expected_macro_place_queue = ["node1", "node2"]
        self.assertEqual(self.placedb.macro_place_queue, expected_macro_place_queue)

if __name__ == '__main__':
    unittest.main()