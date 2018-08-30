"""专门用于各种不同类型算法题研究
"""

import numpy as np
import copy

def test_0():
    """问题描述：任意矩阵中全是0和1构成，且至少有一个0和1。遍历每一个值为1的元素，找到离他最近的0的长度，将
                1更新为这个长度
        示例：
            [[1,0,1],
             [1,1,1],
             [1,1,1]]
        期望结果：
            [[1,0,1],
             [2,1,2],
             [3,2,3]]

        解决方法：使用广度优先遍历，将下一层可以移动的坐标插入队列中，直到某一层遇到0，则一次遍历结束
    """
    def slopOver(x, y, mat):
        """判断是否越界
        """
        if (x < 0 or x >= mat.shape[0]) or \
           (y < 0 or y >= mat.shape[1]):
            return True
        return False

    def findZero(x, y, mat, mat_flag, lst, deep):
        """坐标处是否为0，不为0添加到列表中
        """
        if slopOver(x, y, mat) or \
            mat_flag[x][y] == 1:
            return False

        if mat[x][y] == 0:
            return True
        else:
            mat_flag[x][y] = 1
            lst.insert(0, [x, y, deep+1])
            return False

    # 测试的数组
    mat = np.array([[1,0,1,1,1], [1,1,1,1,1], [1,1,0,1,1], [1,1,1,1,1]])

    m = mat.shape[0]
    n = mat.shape[1]
    size = mat.size

    num = 0 # 第几个元素
    mat1 = copy.deepcopy(mat) # 复制数组作为输出

    while num < size:
        # 得到矩阵下标
        x = num // n
        y = num % n

        # 添加要广度遍历的点
        c_list = []
        c_list.append([x, y, 0])

        # 记录已走过位置的矩阵
        mat_flag = np.zeros(size).reshape(m,n)

        success = False
        print("\nnum=%d, x=%d, y=%d" % (num,x,y))

        while not success:
            print (c_list)
            x_n, y_n, deep = c_list.pop() # 遍历
 
            if mat[x_n][y_n] == 1:
                x_lst = [[x_n-1, y_n], [x_n, y_n-1], [x_n+1, y_n], [x_n, y_n+1]] #生成临近坐标
                for c_i in x_lst:
                    # 发现0则返回，未发现则加入列表
                    if findZero(c_i[0], c_i[1], mat, mat_flag, c_list, deep):
                        deep += 1
                        success = True
                        break
            else:
                success = True
            
            if success == True:
                # 将深度写入矩阵中
                mat1[x][y] = deep
                print(mat1)

        num += 1
    
    print("\nresult:\n",mat1)


if __name__ == '__main__':
    test_0()