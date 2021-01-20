import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import cnnNetwork as myNet # import CNNetwork,CNNetwork_norm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def loadDirData_resize(data_dir):
    # Grayscale()：ImageFolder以RGB三通道形式读取，因此需要转化为灰度图
    transform = transforms.Compose([transforms.Resize((40,40)),   # 数据增强 对应网络结构会变
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    data_set = datasets.ImageFolder(data_dir, transform)
    # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(data_set)
    return data_set


def loadDirData(data_dir):
    # Grayscale()：ImageFolder以RGB三通道形式读取，因此需要转化为灰度图
    transform = transforms.Compose([# transforms.Resize((32,32)),   # 数据增强 对应网络结构会变
        transforms.Grayscale(),
        transforms.ToTensor()])
    data_set = datasets.ImageFolder(data_dir, transform)
    return data_set


'''多网络训练对比'''


def compare_train():
    ############### 参数配置区 ##############
    # 训练相关
    BATCH_SIZE = 32  # 随机梯度batch-size
    WEIGHT_DECAY = 0  # 正则化参数
    MOMENTUM = 0     # 动量
    LEARNING_RATE = 0.1  # 学习率

    # 自定义
    RELOAD_NET = False   # 是否从已有参数文件加载网络
    net_path = ""       # 网络参数加载路径
    num_epochs = 20     # 迭代epoch 次数
    validate_freq = 40   # 验证频率  样本量6240 = n_batch * batch_size

    ##############################################
    ################### 网络对比 ##################
    net = myNet.CNNetwork()
    # net_SGD_norm = CNNetwork_norm()
    net_32 = myNet.CNNetwork_32()
    net_64 = myNet.CNNetwork_64()
    # net_oneFC = myNet.CNNetwork_oneFC()

    # net_Adam = CNNetwork()
    # net_Adagrad = CNNetwork()
    # net_Adamax = CNNetwork()
    nets = [net, net_32, net_64]
    # nets = [net_SGD,net_SGD_norm,net_Momentum,net_Adamax]
    # nets = [net_SGD]

    # 创建不同的优化器用来训练不同的网络
    opt_net = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    opt_net32 = torch.optim.SGD(net_32.parameters(), lr=LEARNING_RATE)
    opt_net64 = torch.optim.SGD(net_64.parameters(), lr=LEARNING_RATE)
    # opt_oneFC = torch.optim.SGD(net_oneFC.parameters(),lr=LEARNING_RATE)

    # opt_SGD_norm = torch.optim.SGD(net_SGD_norm.parameters(),lr=0.1)
    # opt_noPool = torch.optim.SGD(net_noPool.parameters(),lr=LEARNING_RATE,momentum=0.8,nesterov=True)
    # opt_oneConv = torch.optim.SGD(net_oneConv.parameters(),lr=LEARNING_RATE,momentum=0.8,nesterov=True)
    # opt_oneFC = torch.optim.SGD(net_oneFC.parameters(),lr=LEARNING_RATE,momentum=0.8,nesterov=True)

    # opt_Adamax = torch.optim.Adamax(net_Adamax.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)   # 0.002
    # optimizers = [opt_SGD,net_SGD_norm,opt_Momentum,opt_Adamax]
    optimizers = [opt_net, opt_net32, opt_net64]
    # optimizers = [opt_SGD]

    # 'Momentum', 'RMSprop', 'Adam','opt_Adamax']
    labels = ['net', 'net_32', "net_64"]
    losses_list = [[], [], []]

    #################################################
    ###############  实验信息打印 ####################
    print("网络结构试验")
    paraStr = "-[batch]" + str(BATCH_SIZE) + "-[lr]-"+str(LEARNING_RATE) + \
        "[weight-decay]-"+str(WEIGHT_DECAY)+"[momentum]"+str(MOMENTUM)
    print("【参数设置】\n", paraStr)

    # 参数保存目录
    saveDirPath = "C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\cnn-results"

    # 加载训练数据
    train_dir = os.path.join("C:\\Users\\11752\\Desktop\\智能系统\\Lab1", "train")
    train_set = loadDirData(train_dir)
    train_size = len(train_set)
    print("train_size:"+str(train_size))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # 整体加载训练数据，用于验证训练集
    train_eval_loader = DataLoader(train_set, batch_size=train_size)


    # 加载验证数据
    eval_dir = os.path.join(
        "C:\\Users\\11752\\Desktop\\智能系统\\Lab1", "validate")
    eval_set = loadDirData(eval_dir)
    eval_size = len(eval_set)
    print("eval_size:"+str(eval_size))
    eval_loader = DataLoader(eval_set, batch_size=eval_size)

    # 初始化 CNN 网络模型
    # 训练模式
    # 定义优化器

    # 定义学习率控制器
    # stepLR:等间隔（step_size)调整学习率为 gamma 倍
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.9)

    # MultiStepLR: 按照设定间隔调整学习率 （适合观察loss曲线自定义调整
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # 定义loss
    lossFunc = CrossEntropyLoss()


    # 保存最佳参数:通过测试集的loss来衡量效果


    # 画图
    plt.figure("lr compare")
    # plt.title("optimizer compare")


    # 逐epoch训练
    for epoch in range(num_epochs):
        print("======【Epoch %2d Start 】======= " % (epoch + 1))
        batches_loss = 0.0  # validation_freq个batch的平均loss
        # train_correct = 0
        # eval_loss = 0.0
        eval_correct = 0

        # 对每个网络同时遍历 batch
        for i, data in enumerate(train_loader, 0):
            for cnn, optimizer, losses, label in zip(nets, optimizers, losses_list, labels):
                cnn.train()
                # print(label)
                inputs, expects = data
                # 每次迭代都要清零，否则会计算上一次的梯度
                optimizer.zero_grad()
                outputs = cnn(inputs)   # 前向传播
                loss = lossFunc(outputs, expects)   # 计算当前损失
                batches_loss += loss.item()
                loss.backward()  # 反向传播
                optimizer.step()    # 一个batch更新一次参数空间
                # predicts = torch.max(outputs, 1)[1].numpy()
                # train_correct += np.array(predicts == expects.numpy()).sum()
                # 测试验证集
                if (i+1) % validate_freq == 0:
                    eval_loss = 0.0
                    # eval_correct = 0
                    for _, test_data in enumerate(eval_loader, 0):
                        eval_inputs, eval_expects = test_data
                        eval_outputs = cnn(eval_inputs)
                        e_loss = lossFunc(eval_outputs, eval_expects)
                        eval_loss += e_loss.item()
                        # eval_predicts = torch.max(eval_outputs, 1)[1].numpy()
                        # eval_correct += np.array(eval_predicts == eval_expects.numpy()).sum()

                    # 验证模式
                    # batch_loss = loss.item()
                    print(eval_loss)
                    losses.append(eval_loss)
                    # losses.append(batches_loss/validate_freq)
                    batches_loss = 0

            if (i+1) % validate_freq == 0:
                print('epoch:{}/{},steps:{}/{}'.format(epoch+1,
                                                       num_epochs, (i+1)*BATCH_SIZE, train_size))
            # plt.show()

        print('======【Epoch%2d Finished 】======= ' % (epoch + 1))
        # 对每个网络测试验证集结果
        for label, cnn, optimizer in zip(labels, nets, optimizers):
            eval_correct = 0
            for _, data in enumerate(eval_loader, 0):
                eval_inputs, eval_expects = data
                eval_outputs = cnn(eval_inputs)
                eval_predicts = torch.max(eval_outputs, 1)[1].numpy()
                eval_correct += np.array(eval_predicts ==
                                         eval_expects.numpy()).sum()
        # eval_loss /= eval_size
            print(label + ' valid:  [accuracy]:%.2f%% ' %
                  (eval_correct/eval_size*100))

    # 打印个优化器的点
    for j, losses in enumerate(losses_list):
        plt.plot(losses, label=labels[j])
    plt.xlim(0, 80)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(os.path.join(saveDirPath, "网络结构.jpg"))
    plt.show()

    print("训练完成")

''' 单网络训练 '''
def train(resize=False, LEARNING_RATE=0.1):

    ############### 参数配置区 ##############
    # 训练相关
    BATCH_SIZE = 10 # 随机梯度batch-size
    WEIGHT_DECAY = 0.001  # 正则化参数
    MOMENTUM = 0.9  # 0.01     # 动量
    # LEARNING_RATE = 0.1  # 学习率

    # 自定义
    RELOAD_NET = False   # 是否从已有参数文件加载网络
    net_path = ""       # 网络参数加载路径
    num_epochs = 10     # 迭代epoch 次数
    validate_freq = 100   # 验证频率  样本量6240 = n_batch * batch_size

    ##############################################

    ###############  实验信息打印 ####################
    # print("单层fc：120-12")
    paraStr = "-[batch]" + str(BATCH_SIZE) + "-[lr]-"+str(LEARNING_RATE) + \
        "[weight-decay]-"+str(WEIGHT_DECAY)+"[momentum]"+str(MOMENTUM)
    print("【参数设置】\n", paraStr)

    # 参数保存目录
    saveDirPath = "C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\cnn-results"

    # 加载训练数据
    train_dir = os.path.join("C:\\Users\\11752\\Desktop\\智能系统\\Lab1", "train")
    if resize:
        train_set = loadDirData_resize(train_dir)
    else:
        train_set = loadDirData(train_dir)
    train_size = len(train_set)
    print("train_size:"+str(train_size))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # 整体加载训练数据，用于验证训练集
    train_eval_loader = DataLoader(train_set, batch_size=train_size)

    # 加载验证数据
    eval_dir = os.path.join(
        "C:\\Users\\11752\\Desktop\\智能系统\\Lab1", "validate")
    if resize:
        eval_set = loadDirData_resize(eval_dir)
    else:
        eval_set = loadDirData(eval_dir)
    # eval_set = loadDirData(eval_dir)
    eval_size = len(eval_set)
    print("eval_size:"+str(eval_size))
    eval_loader = DataLoader(eval_set, batch_size=eval_size)

    # 初始化 CNN 网络模型
    if resize:
        cnn = myNet.CNNetwork_resize()
    else:
        # cnn = myNet.CNNetwork()
        cnn = myNet.CNNetwork_norm()
    # 训练模式
    cnn.train()
    # 定义优化器
    # optimizer = Adam(network.parameters(), lr=0.07)
    # optimizer = torch.optim.Adamax(cnn.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)   # 0.002
    # print("Adamax:0.002-1")
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # SGD 策略
    optimizer = SGD(cnn.parameters(), lr=LEARNING_RATE,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # optimizer = SGD(cnn.parameters(), lr=LEARNING_RATE,
    #             weight_decay=WEIGHT_DECAY)
    
    # 定义学习率控制器
    # stepLR:等间隔（step_size)调整学习率为 gamma 倍
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.9)

    # MultiStepLR: 按照设定间隔调整学习率 （适合观察loss曲线自定义调整
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # 定义loss
    lossFunc = CrossEntropyLoss()

    # 从已有网络加载参数
    if RELOAD_NET:
        checkpoint = torch.load(net_path)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 保存最佳参数:通过测试集的loss来衡量效果
    best_test_loss = 10
    best_test_accuracy = 0
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    batch_indexes = []

    # 逐epoch训练
    for epoch in range(num_epochs):
        print("======【Epoch %2d Start 】======= " % (epoch + 1))
        batches_loss = 0.0  # validation_freq个batch的平均loss
        train_correct = 0
        eval_loss = 0.0
        eval_correct = 0
        eval_accuracy = 0

        # 遍历 batch
        for i, data in enumerate(train_loader, 0):
            cnn.train()
            inputs, expects = data
            # 每次迭代都要清零，否则会计算上一次的梯度
            optimizer.zero_grad()
            outputs = cnn(inputs)   # 前向传播
            loss = lossFunc(outputs, expects)   # 计算当前损失
            loss.backward()  # 反向传播
            optimizer.step()    # 一个batch更新一次参数空间
            # batches_loss += loss.item()
            predicts = torch.max(outputs, 1)[1].numpy()
            train_correct += np.array(predicts == expects.numpy()).sum()
            # 测试验证集
            if (i+1) % validate_freq == 0:
                # 验证模式
                cnn.eval()
                eval_loss = 0.0
                eval_correct = 0
                batch_loss = loss.item()
                train_losses.append(batch_loss)
                batch_indexes.append(i + 1)
                # 测试验证集结果
                for _, data in enumerate(eval_loader, 0):
                    eval_inputs, eval_expects = data
                    eval_outputs = cnn(eval_inputs)
                    e_loss = lossFunc(eval_outputs, eval_expects)
                    eval_loss += e_loss.item()
                    eval_predicts = torch.max(eval_outputs, 1)[1].numpy()
                    eval_correct += np.array(eval_predicts ==
                                             eval_expects.numpy()).sum()
                # eval_loss /= eval_size
                eval_losses.append(eval_loss)   # 画图数组
                eval_accuracy = eval_correct / eval_size
                eval_accuracies.append(eval_accuracy)
                # 记录验证集loss最佳（最低）时的模型参数
                # if eval_loss < best_test_loss:
                if eval_accuracy > best_test_accuracy:
                    best_test_loss = eval_loss
                    best_test_accuracy = eval_accuracy
                    best_model_state_dict = cnn.state_dict()
                    best_optimizer_state_dict = optimizer.state_dict()

                print('(%2d-%4d)   [Train] Loss:%.6f    [Valid] Loss:%.6f , Accuracy:%.2f%%' % (
                    epoch + 1, i + 1, batch_loss/BATCH_SIZE, eval_loss, eval_accuracy*100))
                batches_loss = 0.0

        # 一个epoch后，检查测试集loss和accuracy
        cnn.eval()
        for _, data in enumerate(train_eval_loader, 0):
            train_eval_inputs, train_eval_expects = data
            train_eval_outputs = cnn(train_eval_inputs)
            te_loss = lossFunc(train_eval_outputs, train_eval_expects)
            train_eval_predicts = torch.max(train_eval_outputs, 1)[1].numpy()
            train_eval_correct = np.array(
                train_eval_predicts == train_eval_expects.numpy()).sum()
            print('======【Epoch%2d Finished 】======= ' % (epoch + 1))
            print('     train: [loss] :%.6f, [accuracy]:%.2f%%\n     valid: [loss] :%.6f, [accuracy]:%.2f%% '
                  % (te_loss.item()/train_size, train_eval_correct/train_size*100, eval_loss, eval_accuracy*100))

    # 画图
    # plt.figure("无pool")
    # plt.title(paraStr)
    # my_x_ticks = np.linspace(1, num_epochs, num_epochs)
    # print(my_x_ticks)
    # plt.plot(train_losses, label="Training loss")

    if resize:
        plt.plot(eval_losses, label='resize')
    else:
        plt.plot(eval_losses, label=LEARNING_RATE)

    # plt.plot(eval_accuracies, label = "eval_accur")
    # plt.xlabel("epoch")
    plt.ylim(0, 2.5)
    # ax = plt.gca()  # 获取当前坐标轴
    # ax.locator_params('x',nbins = num_epochs)
    # plt.xticks(my_x_ticks)
    plt.legend()
    print(os.path.join(saveDirPath, str(best_test_accuracy)+paraStr+".jpg"))
    # plt.savefig(os.path.join(saveDirPath,str(eval_accuracy)+paraStr+"无pool.jpg"))
    # plt.show()

    # 保存模型数据
    log_path = os.path.join(saveDirPath, str(best_test_accuracy)+paraStr)
    print("保存最佳模型...")
    print("验证集loss：", best_test_loss, "验证集accuracy：", best_test_accuracy)
    torch.save({'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': best_optimizer_state_dict,
                'loss': loss},
               log_path)
    print("训练完成")


'''从 data_path 重新加载CNN网络，返回实例'''
def reload_net(data_path):
    model = myNet.CNNetwork_norm()
    checkpoint = torch.load(data_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

''' 逐行打印测试集结果
    test_dir:测试集文件夹
    net_path:网络参数保存文件路径
'''
def test(test_dir, net_path, log_path, size):
    path = test_dir+"\\test"
    file_list = os.listdir(path)
    # 顺序读入，重写文件名补0
    for file in file_list:
        # 补0 10表示补0后名字共10位
        filename = file.zfill(8)
        # print(filename)
        new_name = ''.join(filename)
        os.rename(path + '\\' + file, path + '\\' + new_name)

    test_set = []
    # for imgNum in range(1, size + 1):
    #     # print(imgNum)
    #     img = Image.open(os.path.join(test_dir, str(imgNum) + ".bmp"))
    #     arr = torch.ByteTensor(np.array(img).astype(np.uint8)).view(1,1,28,28)
    #     test_set.append(arr)
    #     # print(data_set[-1])
    # # return data_set

    transform = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])    # 加载测试数据
    test_set = datasets.ImageFolder(test_dir, transform)
    test_size = len(test_set)
    print("test_size:" + str(test_size))
    test_loader = DataLoader(test_set, batch_size=test_size,shuffle=False)
    f = open(log_path, "w")

    # 从参数路径加载网络
    cnn = myNet.CNNetwork_norm()
    checkpoint = torch.load(net_path)
    cnn.load_state_dict(checkpoint['model_state_dict'])
    # 验证模式 （防止改变网络参数）
    cnn.eval()

    index_class_dic = {0:1, 1:10, 2:11, 3:12, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9}
    test_correct = 0
    # 测试验证集结果
    for _, test_data in enumerate(test_loader, 0):
        test_inputs, test_expects = test_data
        test_outputs = cnn(test_inputs)
        # loss = LossFunc(test_outputs, test_expects)
        # test_loss += loss.item()
        predicts = torch.max(test_outputs, 1)[1].numpy()
        # 打印预测结果
        for p in predicts:
            print(index_class_dic[p], file=f)
    # print("验证集loss：",test_loss,"验证集accuracy：",test_correct/test_size)


saveDirPath = "C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\"
# test(test_dir="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\",
#      net_path=saveDirPath+"cnn-results\\0.99375-[batch]10-[lr]-0.01[weight-decay]-0.001[momentum]0.9",log_path="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\cnn_pred.txt",size=1800)
test(test_dir="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\",
     net_path=saveDirPath+"cnn-results\\0.9983333333333333-[batch]10-[lr]-0.01[weight-decay]-0.001[momentum]0.9",log_path="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\cnn_pred.txt",size=1800)
# test(test_dir="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\",
#      net_path=saveDirPath+"cnn-results\\0.99375-[batch]10-[lr]-0.01[weight-decay]-0.001[momentum]0.9",log_path="C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\lab_test\\cnn_pred.txt",size=1800)


def test_save():
    plt.figure("test")
    x = [i for i in range(4)]
    plt.plot(x, x)
    plt.savefig(saveDirPath+"test.jpg")
    plt.show()

# test_save()


saveDirPath = "C:\\Users\\11752\\Desktop\\智能系统\\Lab1\\"

# plt.figure("数据增强")
# # compare_train()
# print("120-120-12")
# train(LEARNING_RATE=0.01)
# train(LEARNING_RATE=0.02)
# train(LEARNING_RATE=0.001)
# # train(resize=True)
# plt.show()
# train()
