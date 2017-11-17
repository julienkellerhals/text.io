import network
import numpy as np

import unittest

class networkTest():

    def conv(self):
        x = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
        n = network.Layer("conv", 3, 3, x)
        r = n.initConv(3, 3, x)
        t = np.array([[4, 3, 4], [2, 4, 3], [2, 3, 4]])

        if r.all() == t.all():
            print("Passed: {} == {}".format(r, t))
        else:
            print("Failed: {} != {}".format(r, t))

    def pool(self):
        x = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]])
        n = network.Layer("pool", 2, prev_layer=x)
        r = n.initPool(2, x)
        t = np.array([[6, 8], [3, 4]])

        if r.all() == t.all():
            print("Passed: {} == {}".format(r, t))
        else:
            print("Failed: {} != {}".format(r, t))
