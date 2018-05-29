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

# cmd = 3
def training_weight_decay(id):
    """规范化，权重衰减"""    
    import training_weight_decay as network

    if (id == 0):
        net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data[:1000], 400, 10, 0.5, 
                evaluation_data = test_data,
                lmbda = 0.1,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True)
    elif (id == 1):
        net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.5, 
                evaluation_data = test_data,
                lmbda = 5,
                # monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_accuracy=True)
                # monitor_training_cost=True)
    elif (id == 2):
        net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 60, 10, 0.1, 
                evaluation_data = validation_data,
                lmbda = 5,
                # monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_accuracy=True)
                # monitor_training_cost=True)

# cmd = 4
def training_initial_weight_with_not_standard_Gaussian_distribution(id = 0):
    """初始化权重为非标准正太分布，权重乘上1/sqrt(n)，年是该神经元对应权重的个数
        偏置仍为标准正太分布。
    """    
    import training_weight_decay as network

    if (id == 0):
        # 初始化权重为标准正太分布
        print '初始化权重为标准正太分布\n'
        net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
        net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.1, 
                evaluation_data = validation_data,
                lmbda = 5.0,
                monitor_evaluation_accuracy=True)
    elif (id == 1):
        # 初始化权重为非标准正太分布
        print '初始化权重为非标准正太分布\n'
        net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
        # net.large_weight_initializer()
        net.SGD(training_data, 30, 10, 0.1,
                evaluation_data = validation_data,
                lmbda = 5.0,
                monitor_evaluation_accuracy=True)
    else:
        print "Invalid cmd id:", id, ', please try again !'

# 主函数
def main(cmd, id):
    """主函数"""
    main_list = (training_init, 
                 training_matrix,
                 training_cross_entropy,
                 training_weight_decay,
                 training_initial_weight_with_not_standard_Gaussian_distribution)
    main_list[cmd](id)

# 主函数
# main(4, 1)

import network2
net = network2.Network([784, 30,10], cost=network2.CrossEntropyCost)
net.SGD(training_data[:10000], 100, 3, 0.1,
        evaluation_data = validation_data[:1000],
        lmbda = 6,
        monitor_evaluation_accuracy=True)