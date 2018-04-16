#!/usr/bin/python
# -*- coding: UTF8 -*-

import mnist_loader

# 读取50000训练数据，10000验证数据，10000测试数据
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

# cmd = 0
def training_init():
    """二次代价常规训练测试"""
    import network
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

# cmd = 1
def training_matrix():
    """小批量训练采用矩阵向量形式"""
    import network_training_matrix as network
    net = network.Network([784, 30, 10])
    # 每个小批量数据越多，矩阵向量反向传播的执行越快
    # SGD_matrix执行反而更慢的原因是组装矩阵时将列表生成矩阵np.array([...])很耗时
    net.SGD(training_data, 1, 5000, 3.0, test_data = None)
    net.SGD_matrix(training_data, 1, 5000, 3.0, test_data = None)

    net.SGD(training_data, 1, 10000, 3.0, test_data = None)
    net.SGD_matrix(training_data, 1, 10000, 3.0, test_data = None)

# cmd = 2
def training_cross_entropy():
    """交叉熵训练测试，使用100个神经元提高了精度"""    
    import network_cross_entropy as network
    net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data = test_data,
    monitor_evaluation_accuracy=True)

# 主函数
def main(cmd_num):
    """主函数"""
    main_list = (training_init, 
                 training_matrix,
                 training_cross_entropy)
    main_list[cmd_num]()

# 主函数
main(0)
