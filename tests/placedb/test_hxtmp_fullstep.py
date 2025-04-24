# tests/placedb/test_hxtmp_fullstep.py
import unittest
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from placedb import HxTmpReader, PlaceDB

class TestHxTmpPlaceDB(unittest.TestCase):

    def setUp(self):
        data_root = os.path.join(PROJECT_ROOT, "benchmark")
        design_name = "blackparrot"
        cache_root = os.path.join(PROJECT_ROOT, "cache")
        self.reader = HxTmpReader(data_root, design_name, cache_root)
        self.placedb = PlaceDB(self.reader)

    def test_design_name(self):
        self.assertEqual(self.placedb.design_name, "blackparrot")

    def test_canvas_size(self):
        self.assertEqual(self.placedb.canvas_size, (600040, 600000))

    def test_hard_macro_num(self):
        self.assertEqual(len(self.placedb.hard_macro_info), 10)

    def test_soft_macro_num(self):
        self.assertEqual(len(self.placedb.soft_macro_info), 0)

    def test_cell_num(self):
        self.assertEqual(len(self.placedb.cell_info), 38180)

    def test_node_num(self):
        self.assertEqual(len(self.placedb.node_info), 38190)

    def test_port_num(self):
        self.assertEqual(len(self.placedb.port_info), 0)

    def test_net_num(self):
        net_ids = [net['id'] for net in self.placedb.net_info.values()]
        net_num = max(net_ids) - min(net_ids) + 1
        self.assertEqual(len(self.placedb.net_info), net_num)

    @unittest.skipIf(sys.platform == "darwin", reason="Skip test_node_data on macOS")
    def test_node_data(self):
        node_file = os.path.join(PROJECT_ROOT, "benchmark", "blackparrot", "zj_node.csv")
        with open(node_file, "r") as f:
            f.readline()
            #inst_id,inst_type,width,height,fixed,x,y,dir
            for line in f:
                blocks = line.strip().split(",")
                inst_id, inst_type, width, height, _fixed, _x, _y, _dir = blocks
                node_name = f"o{inst_id}"
                width = int(width)
                height = int(height)
                self.assertEqual(self.placedb.node_info[node_name]["width"], width)
                self.assertEqual(self.placedb.node_info[node_name]["height"], height)
                if inst_type == "0":
                    self.assertEqual(self.placedb.cell_info[node_name]["width"], width)
                    self.assertEqual(self.placedb.cell_info[node_name]["height"], height)
                else:
                    self.assertEqual(self.placedb.macro_info[node_name]["width"], width)
                    self.assertEqual(self.placedb.macro_info[node_name]["height"], height)
                    self.assertEqual(self.placedb.hard_macro_info[node_name]["width"], width)
                    self.assertEqual(self.placedb.hard_macro_info[node_name]["height"], height)

    @unittest.skipIf(sys.platform == "darwin", reason="Skip test_net_data on macOS")
    def test_net_data(self):
        net_file = os.path.join(PROJECT_ROOT, "benchmark", "blackparrot", "zj_net.csv")
        with open(net_file, "r") as f:
            f.readline()
            #net_id,inst_id,io_type,relative_pin_x,relative_pin_y
            for line in f:
                blocks = line.strip().split(",")
                net_id, inst_id, io_type, relative_pin_x, relative_pin_y = blocks
                net_name = f"n{net_id}"
                node_name = f"o{inst_id}"
                node_type = node_name.upper()
                self.assertIn(net_name, self.placedb.net_info)
                self.assertIn(node_name, self.placedb.net_info[net_name]["nodes"])
                self.assertEqual(node_type, self.placedb.net_info[net_name]["nodes"][node_name]["node_type"])

                width = self.placedb.node_info[node_name]["width"]
                height = self.placedb.node_info[node_name]["height"]
                pin_offset_x = round(float(relative_pin_x) + width / 2)
                pin_offset_y = round(float(relative_pin_y) + height / 2)
                if io_type == "0": # INPUT
                    pin_name = f"I_{pin_offset_x}_{pin_offset_y}"
                else: # OUTPUT
                    pin_name = f"O_{pin_offset_x}_{pin_offset_y}"
                
                self.assertIn(pin_name, self.placedb.net_info[net_name]["nodes"][node_name]["pins"])
                pin_offset = self.placedb.net_info[net_name]["nodes"][node_name]["pins"][pin_name]["pin_offset"]
                self.assertEqual(pin_offset[0], pin_offset_x)
                self.assertEqual(pin_offset[1], pin_offset_y)

    def test_cache_tag(self):
        hard_macro_num = 10
        soft_macro_num = 0
        std_cell_num = 38180
        port_num = 0
        net_num = len(self.placedb.net_info)
        tag = f"{self.placedb.design_name}_H{hard_macro_num}_S{soft_macro_num}_C{std_cell_num}_P{port_num}_N{net_num}"
        self.assertEqual(self.placedb.cache_tag, tag)

if __name__ == "__main__":
    unittest.main()