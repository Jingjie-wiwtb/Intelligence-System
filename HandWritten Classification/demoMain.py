from PIL import Image
import operator
from functools import reduce
import os
import matplotlib.pyplot as plt
import numpy as np
from demo2 import Network
import math

def fit_main():
    # linspace():在某范围产生均匀的数据点
    # 生成训练样本

    input_size = 1000
    input_x = np.linspace(-math.pi, math.pi, input_size)
    input_y = np.zeros((input_size, 1))
    training_set = []

    for i in range(input_size):
        input_y[i] = math.sin(input_x[i])
        training_set.append((input_x[i], input_y[i]))

    plt.figure()
    plt.plot(input_x, input_y)
    # 建立神经网络
    fitNetwork = Network([1,10,1])
    # 训练神经网络
    fitNetwork.SGD(training_set,1000,10,0.1,training_set)

    plt.legend()
    plt.show()


def main():
    bpNetwork = Network([784,32,16,12])
    training_set = load_img("train")
    #
    print("training_set:" + str(len(training_set)))
    #
    validate_set = load_img("validate")
    #
    print("validate_set:" + str(len(validate_set)))
    # print(validate_set)
    #
    # 参数保存目录
    saveDirPath = "C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\classify-test"
    
    evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = bpNetwork.SGD(training_set, 120, 16, 0.05,0,validate_set,True,True,True,True,saveDirPath)
    x = [i for i in range(120)]
    plt.figure("cmd")
    plt.plot(x,evaluation_accuracy,label = 'evaluation_accuracy')
    plt.plot(x,evaluation_cost,label = 'evaluation_cost')
    plt.plot(x,training_cost,label = 'training_cost')
    plt.plot(x,training_accuracy,label = 'training_accuracy')
    plt.legend()
    plt.show()



# 返回 ([1x784],标签) 形式的样本list
# type: 加载类型，可选值："train","validate"，"test"
def load_img(type):
    data_dir = os.path.join("C:\\Users\\11752\\Desktop\\智能系统\\Lab1",type)
    data_set = []
    for i in range(1, 13):
        group_dir = os.path.join(data_dir,str(i))
        expect = np.zeros((12,1))
        expect[i - 1] = 1
        for imgPath in os.listdir(group_dir):
            img = Image.open(os.path.join(group_dir,imgPath))
            # arr = reduce(operator.add,np.array(img).astype(np.uint8).reshape(784,1).tolist())
            arr = np.array(img).astype(np.uint8).reshape(784,1)
            data_set.append((arr, expect))
            
            #print(data_set[-1])
    return data_set

# # print("training_set大小：" + str(len(load_test())))
# print(load_img()[0])

# layers = [2,3,4]
# test = np.array(np.random.randn(row, 1) 
#                         for row in layers[1:])         
# print(test) # 打印不出来，所以最外层还是应该用 list 

# biases = [np.random.randn(row, 1) 
#                         for row in layers[1:]]
# print(biases)

main()