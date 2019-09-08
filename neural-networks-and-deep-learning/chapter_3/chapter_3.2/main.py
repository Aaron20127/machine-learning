#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
import os

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

import mnist_loader
import network2
import command_line
import plot_figure

# 测试函数
def test_0(training_data, validation_data, test_data):
        """使用1000个训练数据，400个迭代周期，验证使用测试数据，观测过渡拟合情况
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data[:1000], 400, 10, 0.5,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


def test_1(training_data, validation_data, test_data):
        """使用50000个训练数据，30个迭代周期，使用测试数据验证，观测过渡拟合
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")

        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_2(training_data, validation_data, test_data):
        """使用1000个训练数据，400个迭代周期，验证使用测试数据，对比test_1()，引入权重衰减，
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")

        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data[:1000], 400, 10, 0.5,
                evaluation_data = test_data,
                lmbda=0.1,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_3(training_data, validation_data, test_data):
        """使用50000个训练数据，30个迭代周期，使用测试数据验证，观测权重衰减
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")

        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                lmbda = 5.0,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_4(training_data, validation_data, test_data):
        """使用50000个训练数据，第二层100个神经元, 30个迭代周期，使用测试数据验证，观测权重衰减
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")

        net = network2.Network([784, 100,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                lmbda = 5.0,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_5(training_data, validation_data, test_data):
        """使用50000个训练数据，60个迭代周期，学习速率为0.1，使用测试数据验证，观测权重衰减
        """
        name = sys._getframe().f_code.co_name
        print(name + "\n")

        net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 60, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

cmd = sys.argv[1:]
# cmd = ['-t', '1']
# main.py -p ta ea -f test_0.net test1.net -a
training_function = [test_0, test_1, test_2, test_3, test_4, test_5]
command_line.register_training_function(training_function)
command_line.execute_command(cmd)






