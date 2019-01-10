　　概念

　　感知机是一种二元线性分类器。输入一组代表实例特征的向量，感知机可以计算出实例的类别。

　　二元分类指的是感知机输出的结果只有两类，代表是或否。实际应用中，一些问题要求的输出就是是或否，比如根据照片识别性别，识别图片中是否存在某种物品，根据X光片判断是否患病，判断邮件是否是垃圾邮件等等。

　　多个是或否的判断嵌套叠加起来，就可以处理复杂的逻辑，也可以输出多元分类。所以，使用多个类似感知机的分类器，组合成一个计算模型，可以解决多种复杂的识别/决策问题。这样的一个计算模型被叫做人工神经网络，其中的单个分类器被叫做人工神经元。

　　数学模型

　　感知机的数学模型是：

　　　　

　　x是代表实例特征的一维数组

　　w是代表每个特征权重的一维数组

　　 是两个数组的点积

　　b是一个代表偏置的常数

　　当时，感知机输出1，代表结果为true，神经元被激活；否则输出0，代表结果为false，神经元未被激活。 

　　几何含义： 这一超平面，把空间分隔成两部分

 

 　　应用示例

　　用感知机来实现并运算：取,,, 对于以下输入通过感知机计算得出y值和并运算真值表的相同，这样我们就通过感知机模拟了一个并运算函数。
 	 	 
 1	0 	0
 0	1	0
 0	0 	0
 1	1	1

 　　下图展示了真值表的坐标，和一条代表感知机的直线。

　　

　　能正确分类的直线有无数条。w和b也有无数种正确的取值。

　　并运算只有四个样本，两个特征，是一个最简单的例子。通常我们要处理的问题有更多样本，更多特征。

　　感知器可以拟合任何的线性函数，可以用来解决任何线性分类。但是解决不了线性不可分的问题，对于线性不可分问题，需要通过多个感知机组成的网络来处理。

　　训练

　　使用给定样本，寻找权重w和偏置b的过程，叫做训练

　　训练算法：

　　　　设置w和b默认值为0，然后不断迭代更新，直到能正确分类所有样本

　　

　　

　　其中

　　

　　

　　t代表训练样本的实际值，被称为label，y代表使用当前参数感知机的输出值，代表学习率，是一个需要手工设定的超参数。

　　下面的代码定义了一个包含训练方法的感知机类型。

    class Perceptron(object):
        def __init__(self, input_num, activator):
            '''
            初始化感知器，设置输入参数的个数，以及激活函数。
            激活函数的类型为double -> double
            '''
            self.activator = activator
            # 权重向量初始化为0
            self.weights = [0.0 for _ in range(input_num)]
            # 偏置项初始化为0
            self.bias = 0.0
        def __str__(self):
            '''
            打印学习到的权重、偏置项
            '''
            return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
        def predict(self, input_vec):
            '''
            输入向量，输出感知器的计算结果
            '''
            # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
            # 变成[(x1,w1),(x2,w2),(x3,w3),...]
            # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
            # 最后利用reduce求和
            return self.activator(
                reduce(lambda a, b: a + b,
                       map(lambda (x, w): x * w,  
                           zip(input_vec, self.weights))
                    , 0.0) + self.bias)
        def train(self, input_vecs, labels, iteration, rate):
            '''
            输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
            '''
            for i in range(iteration):
                self._one_iteration(input_vecs, labels, rate)
        def _one_iteration(self, input_vecs, labels, rate):
            '''
            一次迭代，把所有的训练数据过一遍
            '''
            # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
            # 而每个训练样本是(input_vec, label)
            samples = zip(input_vecs, labels)
            # 对每个样本，按照感知器规则更新权重
            for (input_vec, label) in samples:
                # 计算感知器在当前权重下的输出
                output = self.predict(input_vec)
                # 更新权重
                self._update_weights(input_vec, output, label, rate)
        def _update_weights(self, input_vec, output, label, rate):
            '''
            按照感知器规则更新权重
            '''
            # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
            # 变成[(x1,w1),(x2,w2),(x3,w3),...]
            # 然后利用感知器规则更新权重
            delta = label - output
            self.weights = map(
                lambda (x, w): w + rate * delta * x,
                zip(input_vec, self.weights))
            # 更新bias
            self.bias += rate * delta

 　　使用这个类来训练一个并函数

    def f(x):
        '''
        定义激活函数f
        '''
        return 1 if x > 0 else 0
    def get_training_dataset():
        '''
        基于and真值表构建训练数据
        '''
        # 构建训练数据
        # 输入向量列表
        input_vecs = [[1,1], [0,0], [1,0], [0,1]]
        # 期望的输出列表，注意要与输入一一对应
        # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
        labels = [1, 0, 0, 0]
        return input_vecs, labels    
    def train_and_perceptron():
        '''
        使用and真值表训练感知器
        '''
        # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
        p = Perceptron(2, f)
        # 训练，迭代10轮, 学习速率为0.1
        input_vecs, labels = get_training_dataset()
        p.train(input_vecs, labels, 10, 0.1)
        #返回训练好的感知器
        return p
    if __name__ == '__main__': 
        # 训练and感知器
        and_perception = train_and_perceptron()
        # 打印训练获得的权重
        print and_perception
        # 测试
        print '1 and 1 = %d' % and_perception.predict([1, 1])
        print '0 and 0 = %d' % and_perception.predict([0, 0])
        print '1 and 0 = %d' % and_perception.predict([1, 0])
        print '0 and 1 = %d' % and_perception.predict([0, 1])

 
