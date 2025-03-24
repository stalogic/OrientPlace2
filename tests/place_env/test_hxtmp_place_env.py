# tests/place_env/test_place_env.py
import unittest
import os
import sys
import gym
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from placedb import HxTmpReader, PlaceDB
import place_env

class TestPlaceEnv(unittest.TestCase):

    def setUp(self):
        data_root = os.path.join(PROJECT_ROOT, "benchmark")
        design_name = "blackparrot"
        cache_root = os.path.join(PROJECT_ROOT, "cache")
        reader = HxTmpReader(data_root, design_name, cache_root)
        self.placedb = PlaceDB(reader)

        grid = 224
        self.env = gym.make("orient_env-v0", placedb=self.placedb, grid=grid).unwrapped

    def test_game_step(self):

        self.env.reset()
        done = False
        steps = 0
        while not done:
            orient = self.env.orient_space.sample()
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action, orient)
            steps += 1

        self.assertEqual(done, True)
        self.assertEqual(steps, len(self.placedb.macro_place_queue))

    def test_hpwl_and_cost(self):
        self.env.reset()
        done = False
        while not done:
            orient = self.env.orient_space.sample()
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action, orient)
        hpwl, cost = self.env.calc_hpwl_and_cost()
        self.assertLessEqual(hpwl, cost)

if __name__ == "__main__":
    unittest.main()
