#!/usr/bin/python
# -*- coding: UTF8 -*-

"""对机器学习中一些小的概念的验证，以此加深印象
"""

import sys
sys.path.append("neural-networks-and-deep-learning/common/")

import study_note


class effects_of_different_learning_rates:
    """验证梯度下降中不同的学习速率对学习的影响，可能会导致发散不收敛
       代价函数： f(x) = k*x*x
       导数： f(x) = 2*k*x*x
       收敛： 学习速率大于1将发散，小于1将收敛到0
       扩展思维： 
            (1) 若f(x)有多个波谷，则由于x的初始值不同，将使得梯度下降
                之后x收敛到不同波谷附近。
            (2) 当k=10时，函数比k=1时坡度更陡。坡度越陡，则梯度的模越大，导致在相同的学习效率下，
                梯度下降收敛更快，这也是机器学习中，特征缩放将所有特征缩小在-1~1之间的原因。以只
                有两个特征的线性回归为例。若特征1在0-1000，特征2在0-1，会导致代价函数的等值线不是
                圆形，而是一个椭圆，这样在长轴附近的曲面的就很平缓，因此梯度就会很小，导致训练时
                每次参数的改变量就会很小，收敛缓慢。
                可以参考k=1和k=10时的图figure1的收敛速度，显然k=10是收敛快很多。
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
    effects_of_different_learning_rates().test()