#!/usr/bin/python
# -*- coding: UTF8 -*-

import mnist_loader

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

# training_data = training_data[:9]
# validation_data = validation_data[:9]
# test_data = test_data[:9]    

# print len(training_data), len(validation_data), len(test_data)


import network

# 设置神经网络层数和每层的节点数，这里首层和尾层的节点数只能是784和10
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.5, test_data = test_data)

"""
import network2
net = network.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True)
"""
