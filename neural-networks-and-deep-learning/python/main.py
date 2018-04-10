#!/usr/bin/python
# -*- coding: UTF8 -*-

import mnist_loader

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

# training_data = training_data[:10]
# validation_data = validation_data[:10]
# test_data = test_data[:10]    

# print len(training_data), len(validation_data), len(test_data)

import network

# 设置神经网络层数和每层的节点数，这里首层和尾层的节点数只能是784和10
net = network.Network([784, 30, 10])

# 每个小批量数据越多，矩阵向量反向传播的执行越快
# SGD_matrix执行反而更慢的原因是组装矩阵时将列表生成矩阵np.array([...])很耗时

net.SGD(training_data, 1, 50000, 3.0, test_data = None)
net.SGD_matrix(training_data, 1, 5000, 3.0, test_data = None)

net.SGD(training_data, 1, 10000, 3.0, test_data = None)
net.SGD_matrix(training_data, 1, 10000, 3.0, test_data = None)