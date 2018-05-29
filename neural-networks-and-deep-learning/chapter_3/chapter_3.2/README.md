描述：
这是例子是使用不同数量的训练样本来验证过度拟合的情况，分别是1000和50000个训练样本。代价函数使用的交叉熵，学习速率为0.5，小批量数据为10.

结论：
在训练过程中，训练数据的代价函数持续减小并不意味着测试数据的准确率会越来越高，也不意味着测试数据的代价函数值会越来越小。相反，当训练数据的准确率达到100%时，测试数据的准确率几乎不再变化，而且测试数据的代价函数值出现缓慢增加的情况。然而训练数据的代价函数值会出现持续减小的情况，这显然是一个假想，这种情况就称为过渡拟合了。

.net文件时训练和测试得到的数据，可以直接从训练数据画出图形。