import numpy as np

class Layer:
    def __init__(self):
        pass

    def toArray(self):
        if not self.prediction == None:
            return self.prediction
        else:
            raise ValueError("this layer hasn't been initialized yet")

    def optimize(self, batch, learning_rate):
        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            # find out how wrong our guess was
            delta_del_w, delta_del_b = self.backprop(x, y)

            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_del_w)]
            del_b = [nb+dnb for nb, dnb in zip(del_b, delta_del_b)]

        # update weights and biases based on learning_rate and error
        self.weights = [w-(learning_rate/len(batch))*nw
                         for w, nw in zip(self.weights, del_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                         for b, nb in zip(self.biases, del_b)]

    def backprop(self, x, y):
        del_w = 0
        del_b = 0

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

        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        del_b = delta
        del_w = np.dot(delta, activations[-2].transpose())

        #actual backprop
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            del_b[-l] = delta
            del_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (del_w, del_b)

class Norm(Layer):
    """docstring for Norm"""
    def __init__(self, size):
        self.weights = np.random.randn(size)
        self.biases = np.random.randn(size)
        print("size", size)
        print(self.weights[0])

    def predict(self, prev_layer):
        prev_layer = prev_layer.toArray()

        for w, b in zip(self.weights, self.biases):
            prev_layer = sigmoid(np.dot(w, prev_layer) + b)
        self.prediction = prev_layer
        return (prev_layer)


class Input(object):
    """docstring for Input"""
    def __init__(self, size):
        # no biases
        self.weights = np.random.randn(size)


    def predict(self, input):
        # prev layer has to be defined here already
        self.prev_layer = input

        for w in self.weights:
            prev_layer = sigmoid(np.dot(w, prev_layer) + 0)
        self.prediction = prev_layer
        return (prev_layer)

    def optimize(self, batch, learning_rate):
        del_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            # find out how wrong our guess was
            delta_del_w = self.backprop(x, y)

            del_w = [nw+dnw for nw, dnw in zip(del_w, delta_del_w)]

        # update weights and biases based on learning_rate and error
        self.weights = [w-(learning_rate/len(batch))*nw
                         for w, nw in zip(self.weights, del_w)]

    def backprop(self, x, y):
        del_w = 0

        #set activation
        activation = x
        activations = [x]

        #list for z vectors
        zs = []
        for w in self.weights:
            z = np.dot(w, activation)+0
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        del_w = np.dot(delta, activations[-2].transpose())


        return (del_w)

class Conv(object):
    """docstring for Conv"""
    def __init__(self, size, kernel_size):
        # TODO: all the biases and weights are supposed to be shared?
        self.weights = np.random.randn(size)
        self.biases  = np.random.randn(size)
        #init kernel randomly
        self.kernel  = np.random.randint(-1, 2, (kernel_size, kernel_size))

    def predict(self, prev_layer):
        prev_layer = prev_layer.toArray()
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

        self.prediction = feature_map
        return feature_map

    def optimize(self, batch, learning_rate):
        super.optimize(batch, learning_rate)

        # TODO: update kernel

        
        
class Pool(object):
    """docstring for Pool"""
    def __init__(self, size, pool_size):
        # TODO stride
        self.weights = np.random.randn(size)
        self.biases  = np.random.randn(size)
        self.pool_size = pool_size        

    def predict(self, prev_layer):
        prev_layer = prev_layer.toArray()

        size = self.pool_size
        feature_map = np.zeros((size, size))
        stride = size
        for x in range(0, len(prev_layer)-size+1, stride):
            for y in range(0, len(prev_layer)-size+1, stride):
                # go through the previous layer and take the max in the window
                res = 0
                nums = []
                for i in range(size):
                    for j in range(size):
                        print("{} {}".format(x, y))
                        # add all the numbers to the array so we can take the maxium after
                        nums.append(prev_layer[x+i][y+j])
                feature_map[int(x/size)][int(y/size)] = max(nums)

        self.prediction = feature_map
        return feature_map
        

#sigmoid functions
def sigmoid(p):
    return 1.0/(1.0+np.exp(-p))

def sigmoid_prime(p):
    return sigmoid(p)*(1-sigmoid(p))

# relu function
def relu(x):
    return np.maximum(x, 0)


def cost_derivative(output_activations, z):
        return (output_activations-z)
