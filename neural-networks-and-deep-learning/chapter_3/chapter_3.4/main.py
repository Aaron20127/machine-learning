#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

import mnist_loader
import network2
import command_line
import plot_figure


# 测试函数
def test_0(training_data, validation_data, test_data):
        """使用eta=10， lmbda=1000，训练网络
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 10.0, lmbda=1000.0,
                evaluation_data = validation_data,
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_1(training_data, validation_data, test_data):
        """使用小样本快速的训练
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 10.0, lmbda=1000.0,
                evaluation_data = validation_data[:100],
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_2(training_data, validation_data, test_data):
        """在test_2中使用小样本的基础上，改变参数，快速获得训练的趋势
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 10.0, lmbda=20.0,
                evaluation_data = validation_data[:100],
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_3(training_data, validation_data, test_data):
        """在test_3中使用小样本的基础上，改变参数，使eta=100，发现效果并不是很好
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 100.0, lmbda=20.0,
                evaluation_data = validation_data[:100],
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_4(training_data, validation_data, test_data):
        """在test_4中使用小样本的基础上，改变参数，使eta=1，发现效果变好
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 1.0, lmbda=20.0,
                evaluation_data = validation_data[:100],
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_5(training_data, validation_data, test_data):
        """在test_5中使用小样本的基础上，改变参数，使eta=1，加深神经网络效果更好
        """
        name = sys._getframe().f_code.co_name
        print name + "\n"
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 150, 20, 0.041, lmbda= 0.1,
                evaluation_data = validation_data[:100],
                monitor_training_accuracy=True,
                monitor_training_cost=False,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=False,
                save_figure_feature_file_name = name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


cmd = sys.argv[1:]
training_function = [test_0, test_1, test_2, test_3, test_4, test_5]
command_line.register_training_function(training_function)
command_line.execute_command(cmd)






