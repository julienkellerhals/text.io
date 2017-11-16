import network
import numpy as np

import unittest

class networkTest(unittest.TestCase):
    def __init__(self):
        #nah
        print()

    def setup(self):
        x = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
        n = network.Layer("conv", 3, 3, x)
        r = n.initConv(3, 3, x)
        t = np.array([[4, 3, 4], [2, 4, 3], [2, 3, 4]])

        super.assertEqual(r, t)

    def add(self):
        num = 0
        for _ in range(10):
            num += 1
            for _ in range(2):
                num += 0
        print(num)
