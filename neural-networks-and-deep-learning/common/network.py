#!/usr/bin/python
# -*- coding: UTF8 -*-

"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

class Network(object): 

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        #The number of layers of neurons.
        self.num_layers = len(sizes) 
        #A list of the number of neurons in each layer
        self.sizes = sizes 
        #The biase generated from the second to the last layer.
        #Notice that np.random.randn() has possibility to return negative numbers.
        #Because it returns the x value of the standard positive distribution.
        #从第二层开始，每个神经元有一个偏置
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        # self.biases = [np.linspace(-1, 1, y*1, endpoint=False). \
            # reshape(y, 1) for y in sizes[1:]]     
        #从第二层开始，每一个神经元的权重的数量等于上一层神经元的个数
        #zip()生成元组序列，如zip([1,2],[3,4])输出[(1,3),(2,4)]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # self.weights = [np.linspace(-1, 1, y*x, endpoint=False). \
            # reshape(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """给神经网络一个输入a,返回神经网络的输出"""
        """Return the output of the network if ``a`` is input."""
        # 注意，由于有三层神经网络，则要计算两次输出a，而第二次的输入是第一次的输出a
        # np.dot是矩阵乘法m*n乘上n*z等于m*z，注意是w放在矩阵相乘的前边
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
         test_data=None):
        """training_data 是⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。变量 epochs 和
        mini_batch_size 表示迭代期数量，和采样时的⼩批量数据的⼤⼩，eta 是学习速率。如果给出了test_data，
        则训练完成后将使用test_data给训练后的模型进行测试，这将降低整个运行速度。"""

        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        print 'mini_batch_size:', mini_batch_size
        if test_data: n_test = len(test_data)
        n = len(training_data)
        total = 0.0
        for j in xrange(epochs):
            #训练数据随机排序
            random.shuffle(training_data) 
            #将训练数据每10个一组放入mini_batches，即mini_batches = [[10组],[10组], ...]
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)] 
            #每训练一批数据后得到新的权重和偏置

            time_start = time.time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time_end = time.time()
            execute = time_end - time_start
            # print 'execute: ', execute
            total += execute

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

        print 'total: ', total * 1.0 / epochs, '\n'

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # nabla_b和nabla_b生成和self.biases，self.weights相同类型的值为0的矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        # for循环求出10次代价函数产生的梯度dCx相加，得到一次10个数据训练后的代价函数的梯度
        # 在求self.weights和self.biases需要先将dCx的和除以10
        
        # time_0 = time.time()
        for x, y in mini_batch:
            # 反向传播的算法，⼀种快速计算代价函数的梯度的⽅法，返回每个样本的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 得到一次小批量训练后网络的权重和偏置,b = b - dC * eta,w = w - dC * eta
        # time_1 = time.time()
        # print 'backprop not matrix: ', (time_1 - time_0) #一次小批量反向传播的时间

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # nabla_b和nabla_b生成和self.biases，self.weights相同类型的值为0的矩阵        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        # 1.求出每层的带权输入和输出
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) # 存储每一层带权输入，不包括第一层
            activation = sigmoid(z) # 计算每层激活输出
            activations.append(activation)
        # backward pass
        # 2.求出输出层的误差delta(L)，适应反向传播的BP1函数,delta = (a(L)-y)*(delta'(z(L)))
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # 3.使用BP2求出delta(l),BP3求出偏置的梯度分量和BP4求出权重分量      
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

# sigmoid(z)的导数，做了点变形
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))