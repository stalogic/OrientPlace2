# tests/test_soft_macro.py
import unittest
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from placedb import LefDefReader, build_soft_macro_placedb

class TestSoftMacro(unittest.TestCase):

    def setUp(self):
        data_root = os.path.join(PROJECT_ROOT, "benchmark")
        design_name = "ariane133"
        cache_root = os.path.join(PROJECT_ROOT, "cache")

        self.reader = LefDefReader(data_root, design_name, cache_root)
        self.placedb = build_soft_macro_placedb(self.reader)

    def test_design_name(self):
        self.assertEqual(self.placedb.design_name(), "ariane133")


if __name__ == "__main__":
    # unittest.main()

    data_root = os.path.join(PROJECT_ROOT, "benchmark")
    design_name = "ariane133"
    cache_root = os.path.join(PROJECT_ROOT, "cache")

    reader = LefDefReader(data_root, design_name, cache_root)
    placedb = build_soft_macro_placedb(reader)