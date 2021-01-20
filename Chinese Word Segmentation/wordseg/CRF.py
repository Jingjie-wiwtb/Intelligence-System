from math import inf
from os import pathsep
from os.path import islink
from re import template
import numpy as np
import math
import os
import re
import datetime
import matplotlib.pyplot as plt


states = ['B','I','E','S']
word_set= set()
TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train.utf8"
LABEL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\dataset1\\labels.utf8"
infinite = -3.14e+100   #float(-2.0**31)#负无穷
SAVE_FIG_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\crf_figure"


class CRF_model:
    def __init__(self): #, label_path = LABEL_PATH, train_path = TRAIN_PATH):
        self.label_list = []
        self.U_features = {}
        self.B_features = {}
        self.U_map = {}
        self.B_map = {}
        self.train_list = []
        # self.load_labels(label_path)

    # 模型训练
    def train(self, epoch_num, train_path, evaluate_path, save_dir = None, is_loaded = False):
        print("训练集：")
        train_list = self.read_sentences(train_path)
        print(len(train_list))
        print("验证集：")
        eval_list = self.read_sentences(evaluate_path)
 
        ######## 初始化features_map #########
        if is_loaded == False:
            print("Initialize features...   [",datetime.datetime.now())
            for cnt, sentence in enumerate(train_list):
                for i in range(0, len(sentence)):   # 遍历窗口
                    self.init_features(sentence,i) # 初始化features_map
       
       ######## 训练 ##########
        # 画图
            train_file = os.path.basename(train_path)[0:-5]
            eval_file = os.path.basename(evaluate_path)[0:-5]
            template_file = os.path.basename(self.template_path)[0:-5]
            model_prefix = train_file + "-" + eval_file + "-"+template_file

        else:
            model_prefix = self.load_prefix[:self.load_prefix.find('[')]

        plt.figure(model_prefix)
        train_accu_list = []
        eval_accu_list = []
        best_eval_accu = 0.0
        best_umap = {}
        best_bmap = {}

        print("Start training...   [",datetime.datetime.now())
        for epoch in range(1, epoch_num + 1):
            # train: predict and update
            # 比较 predict 和 ans
            last_train_correct = 0
            train_total = 0
            for sentence in train_list:
                if len(sentence) == 0:
                    continue
                train_total += len(sentence)
                input, expect = zip(*sentence)
                predict = self.predict(input)
                # 遍历，看看哪个不一样就改哪个
                for i, (p, e) in enumerate(zip(predict, expect)):
                    if p != e:
                        # print(input[i],"expect:",e,"predict",p)
                        if i == 0:
                            self.update_features(input, i, type="add", state0=e)
                            self.update_features(input, i, type="sub", state0=p)
                        else:
                            self.update_features(input, i, type="add", state0=e, state1=predict[i-1])
                            self.update_features(input, i, type="sub", state0=p, state1=predict[i-1])
                    else:
                        last_train_correct += 1

            print("Epoch ", epoch)

            # last_train_accu = last_train_correct/train_total*100
            # print("[Epoch",epoch - 1, "]: 训练集：", "{:.2f}".format(last_train_accu), "%")

            train_correct, train_total, _ = self.evaluate(train_list)
            train_accu = train_correct/train_total*100
            train_accu_list.append(train_accu)
            print("[Epoch",epoch, "]: 训练集：", "{:.2f}".format(train_accu), "%")

            eval_correct, eval_total ,_= self.evaluate(eval_list)
            eval_accu =  eval_correct/eval_total * 100
            if eval_accu > best_eval_accu:
                best_eval_accu = eval_accu
                best_umap = self.U_map
                best_bmap =  self.B_map
                self.save_model(save_dir,model_prefix+"["+str(best_eval_accu)+"]",U_map = best_umap, B_map = best_bmap)

            eval_accu_list.append(eval_accu)
            print("[Epoch",epoch, "]: 验证集：","{:.2f}".format(eval_accu),"%")


            # 100% 时停止
            if train_correct == train_total:
                model_prefix += "[accu-{.3f}]".format(best_eval_accu)
                print("训练结束, best eval accu: ", best_eval_accu)
                if save_dir:
                    self.save_model(save_dir,model_prefix,U_map = best_umap, B_map = best_bmap)
                    print("Model saved in ", save_dir, model_prefix)
                    return

        # epoch 结束 保存
        model_prefix += "[accu-{:.3f}]".format(best_eval_accu)
        f = open(os.path.join(save_dir,model_prefix+"eval_accu.txt"),"w")
        f.write(str(eval_accu_list))
        f.close()
        print("训练结束, best eval accu: ", best_eval_accu)
        if save_dir:
            self.save_model(save_dir,model_prefix)
            print("Model saved in ", save_dir, model_prefix)

        # 画图
        plt.plot(train_accu_list, label = 'Train_accu')   
        plt.plot(eval_accu_list, label = 'Eval_accu')   
        plt.legend()
        plt.savefig(os.path.join(SAVE_FIG_PATH, model_prefix +".jpg"))
        print(" Figure saved in ", SAVE_FIG_PATH, model_prefix)
        plt.show()

    # 验证预测值，输出：正确个数/总个数
    def evaluate(self, sentence_list):
        correct = 0
        total = 0   # 总字数
        predict_list = []
        model_prefix = ""
        f = open(os.path.join(SAVE_FIG_PATH,model_prefix+"_Predicts.txt"),"w")
        for sentence in sentence_list:
            if(len(sentence) == 0):
                continue
            total += len(sentence)
            input, expect = zip(*sentence)
            predict = self.predict(input)
            predict_list.append(predict)
            # 遍历，看看哪个不一样就改哪个
            for i, (p, e) in enumerate(zip(predict, expect)):
                if p == e:
                    correct += 1
                else:
                    # f.write(sentence)
                    print(input[i], " ", p," " ,e,file=f)
        f.close()
        return correct, total, predict_list


    # 输入字符串window， 为 cur处state 打分
    def sum_features(self, window, cur, state0, state1 = None):
        u_sum = self.sum_type_features('U', window, cur, state0, state1)
        b_sum = self.sum_type_features('B', window, cur, state0, state1)
        return u_sum + b_sum

    # 输入字符串window， 为 cur处state 打分
    def sum_type_features(self,type, window, cur, state0, state1 = None):
        '''
            此处的 window 是字符串
        '''
        sum = 0
        if type == 'B':
            features = self.B_features
        else:
            features = self.U_features

        for f_name, f_args in features.items():
            key = ""
            for i in range(0, len(f_args)):
                bias = f_args[i][0]
                type = f_args[i][1]

                # if f_name == 'U09':
                #     print(f_name,cur,bias, len(window), window)
                # print(cur + bias < 0 or cur + bias >= len(window))
                if cur + bias < 0 or cur + bias >= len(window):
                    continue
                if type == 0:
                    key +=  window[cur+bias]
                else:   # 1:state
                    if bias == -1:
                        key += state1
                    else:   #0
                        key += state0

            if f_name[0] == 'B':
                f_map = self.B_map[f_name][state0][state1]
            else:
                f_map = self.U_map[f_name][state0]

            if f_map.__contains__(key):
                sum += f_map[key]
        return sum

    # 预测输入的分词标签
    def predict(self, input):
        '''
            根据 feature_map 预测输入的标签
        '''
        V = [{}]
        output = []
        input_len = len(input)
        # print("predict",input)

        max_score = 0
        max_state = ""
        for s in states:
            V[0][s] = self.sum_type_features('U',input[0:3],cur= 0, state0=s)
            if V[0][s] >= max_score:
                max_score = V[0][s]
                max_state = s
        output.append(max_state)
        
        for i in range(1, input_len):
            V.append({})
            max_score = 0
            for state0 in states:
                # print(output[i-1], i, len(output), len(input))
                V[i][state0] = V[i-1][output[i-1]] + self.sum_features(input, i, state0, output[i-1])
                if V[i][state0] > max_score:
                    max_score = V[i][state0]
                    max_state = state0
            output.append(max_state)

        return output
                
    # 初始化所有特征map   
    def init_features(self, window, cur):
        for k,v in self.U_features.items():
            self.init_feature(f_name= k, f_args= v, window=window, cur = cur)

        for k,v in self.B_features.items():
            self.init_feature(f_name= k, f_args= v, window=window, cur = cur)

    # 初始化指定特征函数的map
    def init_feature(self,f_name, f_args, window, cur):
        state0 = window[cur][1]    # 当前字的state
        key = ""
        for i in range(0, len(f_args)):
            bias = f_args[i][0]
            type = f_args[i][1]

            if(cur + bias < 0 or cur + bias >= len(window)):
                return
            key +=  window[cur+bias][type]

        if f_name[0] == 'B':
            state1 = window[cur - 1][1]
            f_map = self.B_map[f_name][state0][state1]
        else:
            f_map = self.U_map[f_name][state0]
        
        if f_map.__contains__(key):
            f_map[key] += 1
        else:
            f_map[key] = 1

    # 更新所有特征map
    def update_features(self, window, cur,type, state0, state1 = None):
        for k,v in self.U_features.items():
            self.update_feature(k,v, window, cur, type, state0,state1)

        for k,v in self.B_features.items():
            self.update_feature(k,v, window, cur, type, state0, state1)


    # 更新指定特征函数的map
    def update_feature(self, f_name, f_args, window, cur, update_type, state0, state1 = None):
        '''
            type: add/ sub
            state: 需要操作的 state
        '''
        # print(cur, window)
        # state0 = window[cur][1]    # 当前字的state
        # state1 = window[cur - 1][1]
        key = ""
        for i in range(0, len(f_args)):
            bias = f_args[i][0]
            type = f_args[i][1]

            if(cur + bias < 0 or cur + bias >= len(window)):
                return
            if type == 0:
                key +=  window[cur+bias]
            else:
                if bias == 0:
                    key += state0
                else:
                    key += state1

        if f_name[0] == 'B':
            if state1:
                f_map = self.B_map[f_name][state0][state1]
            else:
                return
        else:
            f_map = self.U_map[f_name][state0]
        
        if update_type == 'add':
            if not f_map.__contains__(key):
                f_map[key] = 1
            else:
                f_map[key] += 1
        else:   # 'sub
            if not f_map.__contains__(key):
                f_map[key] = -1
            else:
                f_map[key] -= 1

    # 读取样本
    def read_sentences(self, path):
        '''从训练集中读取sentence列表，每个sentence中为（字，state) tuples'''

        states = self.label_list
        f = open(path,"r",encoding="utf-8")
        lines = f.read().splitlines()
        sentence_list = []
        sentence = [] # 记录sentence 的 (字，state) tuples
        for line in lines:
            # 每行由 <observation(字)>  <state> 组成
            str_list = line.split()
            if len(str_list) == 0:
                # new_flag = True# 标记新行
                sentence_list.append(sentence)
                sentence = []
                continue
            # print(str_list)
            character = str_list[0] # 观测值：字
            state = str_list[1]     # 状态：BWES
            sentence.append((character, state))

        print("sentence_size:",len(sentence_list))

        return sentence_list

    # 加载模板文件
    def read_template(self, path):
        ''' return: U_features, B_features'''
        self.template_path = path
        f = open(path,"r", encoding="utf-8")
        U_features = {}
        B_features = {}
        U_map = {}
        B_map = {}
        repat=r'\[-?\d+,-?\d+\]'    #-?[0-9]*

        lines = f.read().splitlines()
        for line in lines:
            line = line.strip()
            # 去除空行和注释行
            if len(line) == 0 or line.find("#") != -1:
                continue

            f_name = line.split(':')[0]
            f_args = []
            for a in list(re.finditer(repat, line)):
                match_str = line[a.start()+1:a.end()-1]
                match = match_str.split(",")
                match = [int(s) for s in match] # 转为数值
                f_args.append(match) 

            if f_name[0] == 'B':
                B_features[f_name] = f_args
                B_map[f_name] = {s:{} for s in states}
                for s1 in states:
                    for s0 in states:
                        B_map[f_name][s1][s0] = {}
                    # for k,v in B_map[f_name][s1].items():
                    #     v = {s0:{} for s0 in states}
            else:
                U_features[f_name] = f_args
                U_map[f_name] = {s:{} for s in states}
        
        # print(U_features)
        # print(B_features)
        self.U_features = U_features
        self.B_features = B_features
        self.U_map = U_map
        self.B_map = B_map
        print("u_features:", len(U_features), ", B_features:", len(B_features))
        
        return U_features, B_features


    # 保存模型
    def save_model(self, save_dir, model_prefix, U_map = None, B_map = None):
        f = open(os.path.join(save_dir,model_prefix+"U_features.txt"),"w")
        f.write(str(self.U_features))
        f.close()

        f = open(os.path.join(save_dir,model_prefix+"U_map.txt"),"w")
        if U_map:
            f.write(str(U_map))
        else:            
            f.write(str(self.U_map))
        f.close()

        f = open(os.path.join(save_dir,model_prefix+"B_features.txt"),"w")
        f.write(str(self.B_features))
        f.close()
        
        f = open(os.path.join(save_dir,model_prefix+"B_map.txt"),"w")
        f.write(str(self.B_map))
        if B_map:
            f.write(str(B_map))
        f.close()
        print("model saved in ", model_prefix)


    # 加载模型
    def load_model(self, save_dir, prefix):
        self.load_prefix = prefix
        f = open(os.path.join(save_dir,prefix+"U_features.txt"),"r")
        self.U_features = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"U_map.txt"),"r")
        self.U_map = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"B_features.txt"),"r",encoding='utf-8')
        self.B_features = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"B_map.txt"),"r")
        self.B_map = eval(f.read())
        f.close()


    # 测试
    def test(self, test_path):
        print("测试样本：",test_path)
        test_list = self.read_sentences(test_path)
        test_correct, test_total, outputs = self.evaluate(test_list)
        # f = open(os.path.join(save_dir,model_prefix+"Predicts.txt"),"w")
        # print(outputs, f)
        test_accu =  test_correct/test_total * 100
        print("正确率：",test_accu,"%")

    # read_template(TEMPLATE_PATH)
# U00:%x[-1,0]
# 前一个字的字
# U00:%x[-1,1]
# 前一个字的state

# 训练主函数
def main():
    # template:
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\test_template.utf8"    # PPT模板(无B)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template2_n1.utf8"    # 网模板（1B:-1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template2_0.utf8"    # 网模板（1B:-1)
    TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template2_1.utf8"    # 网模板（1B:-1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template3.utf8"    # 网模板（无B)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_noss.utf8"    # 网模板（无B)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no-1.utf8"    # 网模板（无-1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no-2.utf8"    # 网模板（无-2)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no1.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no1-3.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no5-8.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_2-2.utf8"    # ppt 
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_2-2noss.utf8"    # ppt 
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no6.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no7.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_no8.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template_win3.utf8"    # 网模板（无1)

    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\win5_no01.utf8"    # 网模板（无1)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\win5_no7.utf8"    # 网模板（无1)

    # train:
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12.utf8"
    TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_30w.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_10w.utf8"

    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12_eval.utf8"
    EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_1w.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_600.utf8"

    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\dataset2\\train.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_500.utf8"

    # save
    SAVE_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\crf_model"
    crf = CRF_model()
    print("train_path:",TRAIN_PATH)
    print("template_path:",TEMPLATE_PATH)
    print("evaluate_path:",EVAL_PATH)
    crf.read_template(TEMPLATE_PATH)
    crf.train(50, TRAIN_PATH,EVAL_PATH,save_dir=SAVE_DIR)

# 测试主函数
def test_main():
    crf = CRF_model()
    # load
    LOAD_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\crf_model"
    # eval:
    TEST_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_500.utf8"
    # TEST_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1.utf8"

    crf.load_model(LOAD_DIR,"train-train1_500-template3[accu-94.134]")  # 这个好nb
    # train1_30w-train2_3w-template_noss[accu-80.473]
    crf.test(TEST_PATH)


def load_main():
    crf = CRF_model()

    # template:
    TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\template3.utf8"    # PPT模板(无B)
    # TEMPLATE_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\test_template.utf8"    # 网模板（无B)

    # load
    LOAD_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\crf_model"
    # LOAD_SURFIX = "train1_total-train2_600-test_template[accu-88.791]"
    # LOAD_SURFIX = "train1_30w-train2_3w-test_template[accu-80.652]"
    # LOAD_SURFIX = "train1_30w-train2_600-template3[accu-85.841]"
    # LOAD_SURFIX = "train1_30w-train2_3w-template3[accu-80.307]"
    LOAD_SURFIX = "train1_30w-train2_3w-template3[accu-86.726]"

    crf.load_model(LOAD_DIR,LOAD_SURFIX)

    # train
    TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_30w.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_total.utf8"

    # eval
    EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_600.utf8"

    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\dataset2\\train.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_500.utf8"

    # save
    SAVE_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\crf_model"
    print("加载模型...")
    print("load_path:",LOAD_SURFIX)
    print("train_path:",TRAIN_PATH)
    print("template_path:",TEMPLATE_PATH)
    print("evaluate_path:",EVAL_PATH)
    # crf.read_template(TEMPLATE_PATH)
    print("继续训练...")
    crf.train(20, TRAIN_PATH,EVAL_PATH,save_dir=SAVE_DIR,is_loaded=True)

# main()
# test_main()
# load_main()