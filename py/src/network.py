import numpy as np
import random
import pickle

#seed our random numbers
np.random.seed(420)


class Layer(object):
    """docstring for Layer."""

    def __init__(self, type, size, kernel_size=None, prev_layer=None):
        self.types = {'norm': 0, 'in': 1, 'conv': 2, 'pool': 3}
        if (type == "norm"):
            # normal layer with weights and biases
            self.type = self.types['norm']
            self.initNorm(size)
        elif (type == "in"):
            # input layer
            self.type = self.types['in']
            self.initIn(size)
        elif (type == "conv"):
            # convolutional layer
            self.type = self.types['conv']
            self.initConv(size, kernel_size, prev_layer)
        elif (type == "pool"):
            # pooling layer
            self.type = self.types['pool']
            self.initPool()
        else:
            raise ValueError("ohno i don't understand this type of layer: {}".format(type))

    def initNorm(self, size):
        # init the weights randomly
        self.weights = [np.random.randn(0, 1) for _ in range(size)]
        self.biases  = [np.random.randn(0, 1) for _ in range(size)]

    def initIn(self, size):
        # no biases
        self.weights = [np.random.randn(0, 1) for _ in range(size)]

    def initConv(self, size, kernel_size, prev_layer):
        # TODO: why kernel_size ?? just size should be enough
        # TODO: all the biases and weights are supposed to be shared?
        self.weights = [np.random.randn(0, 1) for _ in range(size)]
        self.biases  = [np.random.randn(0, 1) for _ in range(size)]
        # TODO: different kenreleleleszszs
        self.kernel  = self.build_kernel(kernel_size)

        self.prev_layer = prev_layer
        length = len(self.kernel)
        feature_map = np.zeros((length, length))
        for x in range(len(prev_layer)-length + 1):
            for y in range(len(prev_layer)-length + 1):
                # go trough the layer with a step size of kernel length
                # print("x: {}, y: {}".format(x, y))
                mini_kernel = 0
                for i in range(length):
                    for j in range(length):
                        if not (x+i >= len(prev_layer) or y+j >= len(prev_layer)):
                            #print("x: {} y: {}".format(x+i, y+j))
                            # print("x+i: {}, y+j: {}".format(x+i, y+j))
                            # multiply each number of the kernel with each part of layer
                            mini_kernel += prev_layer[x+i][y+j] * self.kernel[i][j]

                # rectify
                mini_kernel = relu(mini_kernel)
                feature_map[x][y] = mini_kernel

        self.feature_map = feature_map
        return feature_map

    def initPool(self, size, pool_size, prev_layer):
        self.prev_layer = prev_layer

        feature_map = np.zeros(int(len(prev_layer)/pool_size), int(len(prev_layer)/pool_size))
        stride = pool_size
        for x in range(0, len(prev_layer)-pool_size+1, stride):
            for y in range(0, len(prev_layer)-pool_size+1, stride):
                # go through the previous layer and take the max in the window
                res = 0
                nums = []
                for i in range(pool_size):
                    for j in range(pool_size):
                        # add all the numbers to the array so we can take the maxium after
                        nums.append(prev_layer[x+i][y+j])
                feature_map[int(x/pool_size)][int(y/pool_size)]

        return feature_map

    def build_kernel(self, size):
        """build a kernel"""
        # make empty array
        x = np.zeros((size, size))
        # change middle of array
        x[int(size/2)][int(size/2)] = 1
        x[0][0] = 1
        x[0][size-1] = 1
        x[size-1][0] = 1
        x[size-1][size-1] = 1

        return x

class Network(object):
    """docstring for Network."""
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.initSizes(sizes)

        # initialize weights & biases randomly
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    #def initSizes(self, sizes):
        #for s in sizes:
            #nah

    def conv(self, layer_n, kernel):
        """convolute nth layer with kernel"""
        layer = self.weights[layer_n]
        length = len(kernel)
        res = np.zeros((int(len(layer)/length), int(len(layer)/length)))
        for x in range(0, len(layer), length):
            for y in range(0, len(layer), length):
                # go trough the layer with a step size of kernel length
                mini_kernel = 0
                for i in range(length):
                    for j in range(length):
                        # multiply each number of the kernel with each part of the layer
                        mini_kernel += layer[i][j] * kernel[i][j]
                res[int(x/length)][int(y/length)] = mini_kernel

        return res

    def save(self, path="../../data/net.p"):
        # save the network as a array to a file
        net = np.array([[self.num_layers], [self.sizes], [self.weights], [self.biases]], dtype=object)
        net.dump(path)
        # pickle.dump(net, open(path, "wb"))
        print("saved shit yo")

    def load(self, path="../../data/net.p"):
        # load the network from path
        net = np.load(path)
        # net = pickle.load(open(path, "rb"))
        self.num_layers = net[0][0]
        self.sizes = net[1][0]
        self.weights = net[2][0]
        self.biases = net[3][0]
        # print("loaded shit yo")
        # print("num_layers: ")
        print(self.num_layers)
        return("loaded network")

    def predict(self, x):
        #return output a from input x

        #pickle.dump(x, open("yo.p", "wb"))
        #x = pickle.load(open("py/src/oy.p", "rb"))
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return (x)

    def train(self, data, epochs, num_batches, learning_rate, test_data=None):
        #data is a tuple (input, output)

        for j in range(epochs):
            random.shuffle(data)
            n = len(data)
            batches = [
                data[k:k+num_batches]
                for k in range(0, n, num_batches)
            ]

            #del_w = [np.zeros(w.shape) for w in self.weights]
            #del_b = [np.zeros(b.shape) for b in self.biases]

            for batch in batches:
                #make a prediction with the training data
                #prediction = self.predict(data[0])
                #loss = data[1] - prediction

                self.update_mini_batch(batch, learning_rate)
                #self.weights = [self.weights[y] + learning_rate * loss for y in self.sizes[1:]]

            #print progress
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def update_mini_batch(self, batch, learning_rate):
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            #find out how wrong our guess was
            delta_del_w, delta_del_b = self.backprop(x, y)

            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_del_w)]
            del_b = [nb+dnb for nb, dnb in zip(del_b, delta_del_b)]

        #update weights and biases based on learning_rate and error
        self.weights = [w-(learning_rate/len(batch))*nw
                         for w, nw in zip(self.weights, del_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                         for b, nb in zip(self.biases, del_b)]

    def backprop(self, x, y):
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]

        #set activation
        activation = x
        activations = [x]

        #list for z vectors
        zs = []
        for w, b in zip(self.weights, self.biases):
            #print(len(w))
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].transpose())

        #actual backprop
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (del_w, del_b)

    def cost_derivative(self, output_activations, z):
        return (output_activations-z)

#sigmoid functions
def sigmoid(p):
    return 1.0/(1.0+np.exp(-p))

def sigmoid_prime(p):
    return sigmoid(p)*(1-sigmoid(p))

# relu function
def relu(x):
    return np.maximum(x, 0)
