import network
import numpy as np

def test():
    x = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    n = network.Layer("conv", 3, 3, x)
    x = n.initConv(3, 3, x)
    print(x)
    return x

def add():
    num = 0
    for _ in range(10):
        num += 1
        for _ in range(2):
            num += 0
    print(num)
