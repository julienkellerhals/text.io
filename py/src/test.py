import network
import numpy as np

def test():
  x = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
  n = network.Layer("conv", 3, 3, x)
  print(n)
