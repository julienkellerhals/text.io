import numpy as np
import random
import pickle

#seed our random numbers
np.random.seed(420)

class Network(object):
    """docstring for Network."""
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        #initialize weights & biases randomly
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    def save(self, path="../../data/net.p"):
        #save the network as a array to a file
        net = [self.num_layers, self.sizes, self.weights, self.biases]
        pickle.dump(net, open(path, "wb"))
        print("saved shit yo")

    def load(self, path="../../data/net.p"):
        #load the network from path
        net = pickle.load(open(path, "rb"))
        self.num_layers = net[0]
        self.sizes = net[1]
        self.weights = net[2]
        self.biases = net[3]
        #print("loaded shit yo")
        #print("num_layers: ")
        print(self.num_layers)
        return("loaded network")

    def predict(self, x):
        #return output a from input x

        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

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
        #TODO understand wtf you're doing
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
