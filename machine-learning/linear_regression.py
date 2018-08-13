#!/usr/bin/python
# -*- coding: UTF8 -*-

import sys
import time

sys.path.append("neural-networks-and-deep-learning/common/")

import study_note
import numpy as np
import matplotlib.pyplot as plt


"""
    线性回归分类：
        1.单元线性回归（一个训练数据一个特征）
            h(x) = t0 + t1*x
        2.多元线性回归（一个训练数据多个特征）
            h(x) = t0 + t1*x1 + t2*x2 + ...
        3.多项式回归（一个训练数据一个或多个特征）
            h(x) = t0 + t1*x + t2*x*x + ...

    线性回归解法：
        1.梯度下降法 (单个训练数据特征数量大于10000时适用)
        2.标准方程法 (单个训练数据特征数量小于10000时适用)
        注：多项式回归采用多元线性回归同样的解法
"""

class linear_regression(object):

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.m = x.shape[0]
        self.cost = []

    def fitting_fn(self):
        """拟合函数
        """
        return np.dot(self.x, self.theta) 

    def cost_fn(self):
        """代价函数
        """
        return 1.0/(2*self.m)*\
               sum((self.fitting_fn() - self.y)**2)

    def update_theta(self, alpha):
        """矩阵计算梯度，并更新theta
        """
        self.theta = self.theta - alpha/self.m *\
            np.dot((self.x).T, (np.dot(self.x, self.theta) - self.y))

    def gradient_descent(self, alpha, epoch):
        """梯度下降求参数theta
        """
        for i in range(epoch):
            self.update_theta(alpha)
            cost = self.cost_fn()
            self.cost.append(cost)
            print ("epoch %d, cost %f" % (i, cost))
        print ("theta: \n", self.theta)

        return self.theta, self.cost

    def analytic_method(self):
        """直接根据线性方程组求解theta = (x.T*x).I * x.T * y
        """
        a = np.mat(np.dot(self.x.T, self.x)).I
        self.theta = np.dot(np.array(a), np.dot(self.x.T, self.y))

        cost = self.cost_fn()
        self.cost.append(cost)
        print ("cost: %f" % (cost))
        print ("theta: \n", self.theta)

        return self.theta, self.cost

    def plot_cost(self):
        study_note.plotTest().plot_base(
            y_coordinate = [self.cost],
            title = 'Cost',
            x_lable = 'Epoch',
            y_lable = 'Cost',
            p_type = ['line']) 


### 多元线性回归，欠拟合
def multiple_linear_regression_test():
    """这里只介绍了单变量线性回归，是一个欠拟合曲线情况
       拟合函数y = theta0 + theta1 * x
    """
    def training_data_coordinate():
        """设置需要拟合的坐标
        """
        x_list = np.linspace(0, 1, 8, endpoint=False)
        y_list = np.sqrt(x_list) # 拟合开方的函数
        return x_list, y_list

    def get_training_data():
        """设置训练的点坐标，并生成输入输出矩阵，和theta初始值
           x_array是m*(n+1)维
           y_array是m*1维
           theta是(n+1)*1维
        """
        x_list, y_list = training_data_coordinate()
        # 组合成输入数组
        x_array = np.array([[1, x] for x in x_list]) # 第一列插入1
        y_array = np.array([y_list]).T
        theta = np.array([[1],[1]])

        return x_array, y_array, theta

    def fitting_fn(x, theta):
        """拟合函数方程
        """
        return theta[0] + theta[1] * x

    def plot_fitting_fn(x_start, x_end, theta):
        """绘制拟合曲线
        """
        x_fn = np.linspace(x_start, x_end, 1000, endpoint=False)
        y_fn = [fitting_fn(x, theta) for x in x_fn]

        x_t, y_t = training_data_coordinate()

        study_note.plotTest().plot_base(
            x_coordinate = [x_t, x_fn], 
            y_coordinate = [y_t, y_fn],
            title = 'multiple_linear_regression (underfit)',
            x_lable = 'y',
            y_lable = 'x',
            p_type = ['scatter','line']) 

    # 单一变量x线性回归, 拟合函数y = theta0 + theta1 * x
    x, y, theta = get_training_data()
    method = linear_regression(x, y, theta)
    theta, cost = method.gradient_descent(alpha=0.5, epoch=150) # 梯度下降
    # theta, cost = method.analytic_method() # 解析法

    plot_fitting_fn(0, 1, theta)
    method.plot_cost()
    plt.show()


### 二次多项式回归，拟合
def quadratic_polynomial_regression_test():
    """这里只介绍了单变量线性回归，是一个欠拟合曲线情况
       拟合函数y = theta0 + theta1 * x + theta2 * x * x
    """
    def training_data_coordinate():
        """设置需要拟合的坐标
        """
        x_list = np.linspace(0, 1, 8, endpoint=False)
        y_list = np.sqrt(x_list) # 拟合开方的函数
        return x_list, y_list

    def get_training_data():
        """设置训练的点坐标，并生成输入输出矩阵，和theta初始值
           x_array是m*(n+1)维
           y_array是m*1维
           theta是(n+1)*1维
        """
        x_list, y_list = training_data_coordinate()
        # 组合成输入数组
        x_array = np.array([[1, x, x*x] for x in x_list]) # 第一列插入1,最后一列插入x的平方
        y_array = np.array([y_list]).T
        theta = np.array([[1],[1],[1]])

        return x_array, y_array, theta

    def fitting_fn(x, theta):
        """拟合函数方程
        """
        return theta[0] + theta[1] * x + theta[2] * x * x

    def plot_fitting_fn(x_start, x_end, theta):
        """绘制拟合曲线
        """
        x_fn = np.linspace(x_start, x_end, 1000, endpoint=False)
        y_fn = [fitting_fn(x, theta) for x in x_fn]

        x_t, y_t = training_data_coordinate()

        study_note.plotTest().plot_base(
            x_coordinate = [x_t, x_fn], 
            y_coordinate = [y_t, y_fn],
            title = 'quadratic_polynomial_regression (fit)',
            x_lable = 'y',
            y_lable = 'x',
            p_type = ['scatter','line']) 


    x, y, theta = get_training_data()
    method = linear_regression(x, y, theta)
    # theta, cost = method.gradient_descent(alpha=0.5, epoch=3200) # 梯度下降
    theta, cost = method.analytic_method() # 解析法

    plot_fitting_fn(0, 1, theta)
    method.plot_cost()
    plt.show()


### 四次多项式回归，过拟合
def quartic_polynomial_regression_test():
    """这里只介绍了单变量线性回归，是一个欠拟合曲线情况
       拟合函数y = theta0 + theta1 *x + theta2 *x^2 + theta3 *x^3 + theta4 *x^4
    """
    def training_data_coordinate():
        """设置需要拟合的坐标
        """
        x_list = np.linspace(0, 1, 8, endpoint=False)
        y_list = np.sqrt(x_list) # 拟合开方的函数
        return x_list, y_list

    def get_training_data():
        """设置训练的点坐标，并生成输入输出矩阵，和theta初始值
           x_array是m*(n+1)维
           y_array是m*1维
           theta是(n+1)*1维
        """
        x_list, y_list = training_data_coordinate()
        # 组合成输入数组
        x_array = np.array([[1, x, x*x, x*x*x, x*x*x*x] for x in x_list]) # 插入特征
        y_array = np.array([y_list]).T
        theta = np.array([[1],[1],[1],[1],[1]])

        return x_array, y_array, theta

    def fitting_fn(x, theta):
        """拟合函数方程
        """
        return theta[0] + theta[1]*x + theta[2]*x*x + \
                theta[3]*x*x*x + theta[4]*x*x*x*x

    def plot_fitting_fn(x_start, x_end, theta):
        """绘制拟合曲线
        """
        x_fn = np.linspace(x_start, x_end, 1000, endpoint=False)
        y_fn = [fitting_fn(x, theta) for x in x_fn]

        x_t, y_t = training_data_coordinate()

        study_note.plotTest().plot_base(
            x_coordinate = [x_t, x_fn], 
            y_coordinate = [y_t, y_fn],
            title = 'quartic_polynomial_regression (overfit)',
            x_lable = 'y',
            y_lable = 'x',
            p_type = ['scatter','line']) 

    x, y, theta = get_training_data()
    method = linear_regression(x, y, theta)
    # theta, cost = method.gradient_descent(alpha=1, epoch=2000000) # 梯度下降
    theta, cost = method.analytic_method() # 解析法

    plot_fitting_fn(0, 1, theta)
    method.plot_cost()
    plt.show()

if __name__=="__main__":
    ## 一元线性回归，欠拟合
    # multiple_linear_regression_test()

    ## 一元二阶多项式线性回归，拟合 
    # quadratic_polynomial_regression_test()

    ## 一元四阶多项式线性回归，过拟合 
    # quartic_polynomial_regression_test()
    print ("\n")
    print (os.environ)
    print ("\n")
        