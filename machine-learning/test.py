#!/usr/bin/python
# -*- coding: UTF8 -*-

import sys
sys.path.append("../neural-networks-and-deep-learning/common/")

import study_note


class simple_gradient_descent_test:
    """对简单代价函数使用梯度下降
       代价函数： f(x) = k*x*x
       导数： f(x) = 2*k*x*x
       收敛： 学习速率大于1将发散，小于1将收敛到0
       扩展思维： 若f(x)有多个波谷，则由于x的初始值不同，将使得梯度下降
                之后x收敛到不同波谷附近，前提是学习速率足够小
    """

    def derivative_fun(self, k, x): 
        return 2.0*k*x

    def gradient_descent(self, k, x_start, epochs, alpha):
        x_set = []
        x1 = x_start
        for i in range(epochs):
            x_set.append(x1)
            x1 = x1 - alpha * self.derivative_fun(k, x1)
        return x_set

    def test(self):
        k = 1
        x_start = 10.0 # x的开始位置
        epochs = 500 # 下降周期
        alpha_list = [0.01, 0.1, 0.3, 0.6, 0.99, 1, 1.01] #学习速率列表

        for alpha in alpha_list:
            x_set = self.gradient_descent(k, x_start, epochs, alpha)
            study_note.plotTest().plot_base(
                    [x_set],
                    title = 'alpha = '+str(alpha),
                    x_lable = 'Epoch',
                    y_lable = 'x') 

        study_note.plt.show()

if __name__=="__main__":
    simple_gradient_descent_test().test()