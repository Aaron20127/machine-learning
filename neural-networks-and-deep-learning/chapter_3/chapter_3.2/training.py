#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import sys
sys.path.append("../../common")

import mnist_loader
import network2

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 测试函数
def test_1():
        """使用1000个训练数据，400个迭代周期，验证使用测试数据，观测过渡拟合情况
        """
        print "test_1\n"
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data[:1000], 400, 10, 0.5,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name='test_1.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=True)


def test_2():
        """使用50000个训练数据，30个迭代周期，使用测试数据验证，观测过渡拟合
        """
        print "test_2\n"
        
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name='test_2.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=True)

def test_3():
        """使用1000个训练数据，400个迭代周期，验证使用测试数据，对比test_1()，引入权重衰减，
        """
        print "test_3\n"
        
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data[:1000], 400, 10, 0.5,
                evaluation_data = test_data,
                lmbda=0.1,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name='test_3.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=True)

def test_4():
        """使用50000个训练数据，30个迭代周期，使用测试数据验证，观测权重衰减
        """
        print "test_4\n"
        
        net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                lmbda = 5.0,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name='test_4.net',
                B_plot_figure_feature=False,
                B_show_figure_feature=True)

def test_5():
        """使用50000个训练数据，30个迭代周期，使用测试数据验证，观测权重衰减
        """
        print "test_5\n"
        
        net = network2.Network([784, 100,10], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5,
                lmbda = 5.0,
                evaluation_data = test_data,
                monitor_training_accuracy=True,
                monitor_training_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                save_figure_feature_file_name='test_5.net',
                B_plot_figure_feature=True,
                B_show_figure_feature=True)

### python training 1 执行第一个训练

def main(arg):
        cmd = [0, test_1, test_2, test_3, test_4, test_5]
        cmd[int(arg)]()
                

main(sys.argv[1])






