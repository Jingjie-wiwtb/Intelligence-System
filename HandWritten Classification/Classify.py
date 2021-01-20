# numpy 是一个支持大量N维数组和矩阵运算的数学函数库
from math import exp
import numpy as np
import os
from numpy import random
from numpy.core.defchararray import translate



# 基于随机梯度下降学习

class Network(object):
    # layers: 每层包含的神经元个数
    # layer_cnt: 总层数
    # biases：每层bias列向量 的数组
    # weights：每层权重矩阵 的数组 
    # rate: 学习率
    '''
    (e.g.: 第L-1层3个神经元，第L层2个神经元,
        L-1和L层之间的权重矩阵为：weights[L - 2] = [(w11,w12,w13),(w21,w22,w23)],维度 layers[L-1] x layers[L-2]
        第L层的偏移列向量为：bias[L - 2] = [(b1),(b2)]
        第L层的加权和为：z[L - 1] = weights[L - 2]*z[L - 2] + biases[L - 2]
    '''
    def __init__(self, layers, init_bias = 1, init_weight = 1):
        self.layers = layers
        self.layer_cnt = len(layers)
        # 初始化偏移
        self.biases = [np.random.randn(row, 1) * init_bias 
                        for row in layers[1:]]    # 对隐藏层和输出层的每个节点初始化一个bias[节点数x1]列向量， 1表示输入层没有bias 
        # print(self.biases)
        self.weights = [np.random.randn(row, line) * init_weight
                        for row, line in zip( layers[1:], layers[:-1])]
        # print(self.weights)
        '''
            假设 layers = [3,2,4]
            weights = [ W12[2x3], W23[4x2]]
            layers[1:] = [2,4],layers[:-1] = [3,2,4]
            zip(layers[1:],layers[:-1]) = [(2,3),(4,2)]
        '''


    # 前向传播
    # input: [输入层神经元个数 x 样本个数]
    def feed_forward(self, input):
        ''' 逐层计算下一层的输出，返回最终的输出层结果 '''
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid(np.dot(weight, input) + bias)
        return input


    '''
    梯度下降的分类：
        【batch梯度下降】   
            将所有训练数据输入学习
        【随机梯度下降】
            每次只随机输入一个样本进行学习
        【mini-batch梯度下降】（批梯度下降）
            前两个方法的折中，将训练数据分成若干小份，每次训练一份,(计算这一小批量的梯度然后更新，很好的防止过拟合的办法)
        
        一个像素点表示一个输入层神经元，输入为28*28=784的向量
        对每个训练样本的每个输入神经元都要计算梯度，样本较多时，会出现性能问题

        【关于batch-size 的选择】
        随着 batch_size 增大，处理相同数据量的速度加快，达到相同精度所需要的epoch数量变多，
        结合两种因素的矛盾，batch_size增加到一定程度时达到最优
        内存利用率高，大矩阵乘法并行化效率提高、跑完一次epoch（全数据集）所需迭代次数减少
    '''
    # 随机梯度下降法
    def stochastic_gradient_descent(self, training_set, epochs, batch_size,rate, validation_set = None ):
        # training_data: tuple(input, expect_output)的list 表示输入和期望输出
        # mini_batch: 每次输入的训练集大小
        # epochs：迭代次数，一个epoch等于使用徐连集中的全部样本训练一次 （辨析: iteration: 一个iteration等于使用batch_size个样本训练一次
        # rate: 学习速率
        # validation_set: 默认没有，如果给了验证集，每次迭代之后都验证一次正确率并打印
        if validation_set: 
            validation_size = len(validation_set)
        else:
            validation_size = 0

        train_size = len(training_set)

        validate_interval = 100
        print_cnt = validate_interval
        for epoch in range(epochs):
            print_cnt -= 1
            # 先打乱样本,将测试数据集划分为大小为 batch_size 的小样本集
            random.shuffle(training_set)
            # 划分的小批量样本集的数组 
            batches = [
                training_set[k:k + batch_size] for k in range(0, train_size, batch_size)
            ]
            # print("batch size：%d" % len(batches))
            batchCnt = 0
            for batch in batches:
                batchCnt += 1
                self.update_parameter(batch, rate)

            if print_cnt == 0 and validation_set:
                print_cnt = validate_interval
                correct_size = self.evaluate_fit(validation_set)
                print("Epoch {0}: {1} / {2}, 正确率：{3}".format(
                    epoch, correct_size, validation_size, correct_size/validation_size)
                )
                # save(str(epoch).join("-weight.txt"), np.array(self.weights))

            else:
                print("Epoch {0} complete".format(epoch))



    # 用一个批量的训练参数更新该神经网络的参数 (weights、bias)
    # 更新策略：梯度下降 + 反向传播
    def update_parameter(self, batch, rate):
        # numpy ndarray 版本
        # delta_b = np.array(np.zeros(b.shape) for b in self.biases)
        # delta_w = np.array(np.zeros(w.shape) for w in self.weights)
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        # 遍历训练样本，每次更新参数
        for input, expect in batch: # input是一个[784x1], expect是一个数
            # bias 和 weight 的更新量
            next_delta_b, next_delta_w = self.back_propogation(input, expect)
            # print(next_delta_b)
            # 因为是矩阵，不能用 +=
            delta_b = [nb + dnb for nb, dnb in zip(delta_b, next_delta_b)]
            delta_w = [nw + dnw for nw, dnw in zip(delta_w, next_delta_w)]
            # delta_b += next_delta_b
            # delta_w += next_delta_w

        # 把总体梯度转换为随机选取的mini-batch 的梯度
        # 更新偏移 b' = b - (rate/m)* ∂C/∂b
        self.biases = [b - (rate/len(batch)) * nb
                            for b, nb in zip(self.biases, delta_b)]
        # 更新权重 w' = w - (rate/m)* ∂C/∂w
        self.weights = [w - (rate/len(batch)) * nw 
                            for w, nw in zip(self.weights, delta_w)]
        # self.biases -= (rate/len(batch))*delta_b
        # self.weights -= (rate/len(batch))*delta_w
        


    # 验证函数
    # 统计输出和期望值相等的数量
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(input)), np.argmax(expect))   # argmax()返回最大值的索引，激活值最大的索引即为预测的分类索引
                            for (input,expect) in test_data]
        # print(test_results)
        return sum(int(output == expect) for (output, expect) in test_results)

    def evaluate_fit(self,test_data):
        return sum(int(output == expect) for (output, expect) in test_data)
    '''
        损失函数 VS 代价函数
        【损失函数】 对于单个样本的损失或误差
        【代价函数】 多样本同时输入的总体误差（单个样本误差和的平均值）
    '''
    # 代价函数的导数 
    # 此处采用二次代价函数 (单样本loss_func: 1/2(expect - output)^2
    def cost_derivative(self, output, expect):
        return (output - expect)

    # 反向传播算法
    '''
        【符号约定】
            W(jk)[L] 第(l-1)层第k个神经元指向第L层第j个神经元的权重
            线性加权结果 z
            激活结果(activation): σ 
        delta:
            为每个神经元定义一个误差
            delta = ∂E/∂zⱼ = ∂E/∂a * ∂a/∂z (lossFunc对该神经元的输入(上一层的加权和）的偏导数
            反映该神经元的输入对代价函数的影响, 该影响和上一层的权重矩阵反向传播到上一层
        bias调整量：
            代价函数对偏移量的偏导 = delta
        weight调整量：
            代价函数对weight的偏导 = delta * 上一层的激活结果
    '''
    # 返回 bias 和 weights 的调整量
    def back_propogation(self, input, expect):
        # 生成和self.biases、self.weights 相同形状的 0矩阵
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = input
        activation_list = [input] # 逐层储存所有激活结果的数组 (一行一层，因为要append)
        z_list = [] # 逐层储存所有加权和的数组

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)


        #backward pass
        '''delta = ∂E/∂zⱼ = ∂E/∂a * ∂a/∂z '''
        # 从最后一层向前计算误差，对于二次代价函数：δ = (output - y) * σ'(z)
        delta = self.cost_derivative(activation_list[-1], expect) * sigmoid_derivative(z_list[-1])
        '''bias调整量：
            代价函数对偏移量的偏导 = delta '''
        delta_b[-1] = delta
        ''' weight调整量：
            代价函数对weight的偏导 = delta · 上一层的激活结果'''
        delta_w[-1] = np.dot(delta, activation_list[-2].transpose())    # transpose: 数组的转置

        # 此处的layer从后往前
        # 从倒数第一层开始 (-2+1=-1)
        for layer in range(2, self.layer_cnt):
            z = z_list[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_derivative(z)
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, activation_list[-layer-1].transpose())
        
        return (delta_b, delta_w)

    
    def save(self,weightPath, biasPath):
        np.savetxt(weightPath,self.weights,'%s')
        np.savetxt(biasPath,self.biases,'%s')


    def load(self,weightPath, biasPath):
        self.biases = np.loadtxt(biasPath)
        self.weights = np.loadtxt(weightPath)


# 激活函数 sigmoid (向量运算)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# sigmoid 函数的导数
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# def normalize(x):
