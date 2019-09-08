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


### 不同深度导致的学习效率不同，并不是网络越深效果越好
def test_0(training_data, validation_data, test_data):
        """ 一个隐藏层，对比不同深度之间的效果
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_1(training_data, validation_data, test_data):
        """两个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_2(training_data, validation_data, test_data):
        """3个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


def test_3(training_data, validation_data, test_data):
        """4个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)





### 4-7，使用少量数据观察偏置梯度的变化，不使用minibach，因为minibatch会给偏置带来噪声

def test_4(training_data, validation_data, test_data):
        """ 一个隐藏层，对比不同深度之间的效果
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_5(training_data, validation_data, test_data):
        """两个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_6(training_data, validation_data, test_data):
        """3个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)


def test_7(training_data, validation_data, test_data):
        """4个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 500, 1000, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)







### 使用minibach 后梯度变换的情况

def test_8(training_data, validation_data, test_data):
        """ 一个隐藏层，对比不同深度之间的效果
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_9(training_data, validation_data, test_data):
        """两个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

def test_10(training_data, validation_data, test_data):
        """3个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)



def test_11(training_data, validation_data, test_data):
        """4个隐藏层
        """
        name = sys._getframe().f_code.co_name
        print (name + "\n")
        net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data[:1000], 30, 10, 0.1,
                lmbda = 5.0,
                evaluation_data = validation_data,
                # monitor_training_accuracy=True,
                # monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                # monitor_evaluation_cost=True,
                save_figure_feature_file_name=name + '.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=False)

cmd = ['-t', "0"] 
# cmd = ['-p', 'bg1', 'bgo', '-f', 'test_0.net', 'test_1.net', '-e']
# cmd = ['-p', 'bg1', 'bgo', '-f', 'test_0.net', 'test_1.net', '-a']

# cmd = sys.argv[1:]
training_function = [test_0, test_1, test_2, test_3,
                     test_4, test_5, test_6, test_7,
                     test_8, test_9, test_10, test_11]

command_line.register_training_function(training_function)
command_line.execute_command(cmd)






