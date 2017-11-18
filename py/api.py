from __future__ import print_function
from calc import calc as real_calc
from src import network
from src import my_mnist_loader
from src.layers import *
import sys
import zerorpc
import numpy as np

class Api(object):
    def calc(self, text):
        """based on the input text, return the int result"""
        try:
            return real_calc(text)
        except Exception as e:
            return 0.0

    def echo(self, text):
        """echo any text"""
        self.network = None
        return text

    def init_network(self):
        """init a empy network with set sizes"""
        self.net = network.Network()
        self.net.add("in", 784)
        self.net.add("conv", 372, kernel_size=7)
        self.net.add("pool", 186, pool_size=7)
        self.net.add("conv", 186, kernel_size=5)
        self.net.add("pool", 93, pool_size=5)
        self.net.add("norm", 40)
        self.net.add("norm", 10) # output layer
        return("made a network yo")

    def load_network(self, path=None):
        """load network from path"""
        if (self.net == None):
            return("Initialize the network first")
        else:
            if (path == None):
                return(self.net.load())
            else:
                return(self.net.load(path))

    def save_network(self, path=None):
        """save network to path"""
        if (self.net == None):
            return("Initialize the network first")
        else:
            if (path == None):
                return(self.net.save())
            else:
                return(self.net.save(path))

    def train(self, data, epochs, num_batches, learning_rate, test_data=None):
        """train the network"""
        if (self.net == None):
            return("Initialize the network first")
        else:
            return(self.net.train(data, epochs, num_batches, learning_rate, test_data))

    def predict(self, x):
        """predict array x"""
        if (self.net == None):
            return("Initialize the network first")
        else:
            #TODO: CHANGE THIS TO LENGHT OF FIRST LAYER
            #turn x into a numpy array that we can use without getting grabage as result
            x = np.array(x, dtype=float)
            x.shape = (784, 1)
            x = self.net.predict(x)
            res = np.argmax(x)
            #we can't return a numpy class so we need .item()
            return(res.item())

def parse_port():
    port = 4242
    try:
        port = int(sys.argv[1])
    except Exception as e:
        pass
    return '{}'.format(port)

def main():
    addr = 'tcp://127.0.0.1:' + parse_port()
    s = zerorpc.Server(Api())
    s.bind(addr)
    print('start running on {}'.format(addr))
    s.run()

if __name__ == '__main__':
    main()
