#!/usr/bin/python
# -*- coding: UTF8 -*-

"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
"""

#### Libraries
# Standard library
import threading
import math
import json
import random
import sys
sys.path.append("../common")

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

import base_module as bml
import plot_figure
#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, momentum = False):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        # 画图数据初始化
        self.figure_feature = {
            "title" : {
            },
            "result" : [
            ]
        }     
        
        self.figure_feature["title"]["sizes"] = sizes

        if cost == CrossEntropyCost:
            self.figure_feature["title"]["cost"] = "CrossEntropyCost"
        elif cost == QuadraticCost:
            self.figure_feature["title"]["cost"] = "QuadraticCost"
        else:
            self.figure_feature["title"]["cost"] = ""    

        # 私有数据初始化
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost=cost
        self.default_weight_initializer()
        self.evaluation_cost = []
        self.evaluation_accuracy = []
        self.training_cost = []
        self.training_accuracy = []

        # 每一层偏置的梯度的模
        self.bias_gradient_norms = [] 
        for i in range(self.num_layers - 1):
            self.bias_gradient_norms.append([])

        self.monmentum = momentum
        if momentum:
            ### 增加momentum，vb，vw表示运动速度
            self.vb = [np.zeros((y, 1)) for y in self.sizes[1:]]
            self.vw = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            self.u = 0.9 #摩擦系数

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # np.random.randn(y, 1), 返回array, shape(y,1)的标准正太分布
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.figure_feature["title"]["weight"] = "default"        

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.figure_feature["title"]["weight"] = "large"

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            c = np.dot(w, a)
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            save_figure_feature_file_name = None,
            B_plot_figure_feature = False,
            B_show_figure_feature = False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        self.figure_feature["title"]["training"] = len(training_data) 
        self.figure_feature["title"]["evaluation"] = len(evaluation_data)   
        self.figure_feature["title"]["epochs"] = epochs
        self.figure_feature["title"]["batch_size"] = mini_batch_size
        self.figure_feature["title"]["eta"] = eta
        self.figure_feature["title"]["lmbda"] = lmbda

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                self.training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                self.training_accuracy.append(accuracy * 1.0 / n)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                self.evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                self.evaluation_accuracy.append(accuracy * 1.0 / n_data)
                print ("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))

            print ("\n")
        
        # 保存运算结果
        data_list = [
            ["training cost", self.training_cost, 'cost'],
            ["training accuracy", self.training_accuracy, 'accuracy'], 
            ["evaluation cost", self.evaluation_cost, 'cost'], 
            ["evaluation accuracy", self.evaluation_accuracy, 'accuracy']
        ]
        
        # 保存隐藏层偏置梯度的模
        for i in range(len(self.bias_gradient_norms[:-1])):
            data = ['hidden layer %s bias gradient' % (i+1),
                     self.bias_gradient_norms[i],
                    'bias gradient']
            data_list.append(data)

        # 保存输出层偏置梯度的模
        data = ['output layer bias gradient', 
                 self.bias_gradient_norms[-1],
                'bias gradient']
        data_list.append(data)

        result = self.figure_feature["result"] 
        for i in range(len(data_list)):
            data_obj = {}
            if data_list[i][1]:
                data_obj["name"] = data_list[i][0]    
                data_obj["type"] = data_list[i][2]                
                self.generate_result_parameter(data_obj, data_list[i][1]) 
                result.append(data_obj)    

        # 打印保存的数据
        if B_show_figure_feature:
            self.print_figure_feature()

        # 保存数据
        if save_figure_feature_file_name:
            self.save_figure_feature(save_figure_feature_file_name)

        return self.evaluation_cost, self.evaluation_accuracy, \
            self.training_cost, self.training_accuracy

    def print_figure_feature(self):
        """打印训练数据
        """
        print (json.dumps(self.figure_feature))

    def generate_result_parameter(self, data_obj, data):
        data_obj["data"] = data
        data_obj["min"] = [data.index(min(data)), float('%.6f' % min(data))]
        data_obj["max"] = [data.index(max(data)), float('%.6f' % max(data))]

    def save_figure_feature(self, file):
        """保存训练数据到文件
        """
        bml.write_list_to_file(file, self.figure_feature)

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 记录每次mini batch的偏置的平均梯度值，梯度值只记录隐藏层
        b_average_gradient = [(np.reshape(nb, len(nb))/len(mini_batch)).tolist() for nb in nabla_b]
        b_norms = [list_norm(avg) for avg in b_average_gradient]

        for i in range(len(b_norms)):
            self.bias_gradient_norms[i].append(b_norms[i])

        if self.monmentum:
            ### monmentum 梯度下降方法
            self.vw = [self.u * vw - eta*(lmbda/n)*w-(eta/len(mini_batch))*nw
                            for vw, w, nw in zip(self.vw, self.weights, nabla_w)]
            self.vb = [self.u * vb - (eta/len(mini_batch))*nb
                            for vb, nb in zip(self.vb, nabla_b)]

            self.weights = [w + vw
                            for w, vw in zip(self.weights, self.vw)]
            self.biases  = [b + vb
                            for b, vb in zip(self.biases, self.vb)]
        else:
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        # 交叉熵的L2规范化代价函数求代价
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# 求向量的模，向量由list表示
def list_norm(l):
    return math.sqrt(sum([x*x for x in l]))