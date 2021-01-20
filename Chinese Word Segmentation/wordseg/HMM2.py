from math import inf
from os import pathsep
import numpy as np
import math
import os


state_list = ['B','M','E','S']
TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12.utf8"
LABEL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\dataset2\\labels.utf8"
infinite = -3.14e+100   #float(-2.0**31)#负无穷


class HmmModel:

    def __init__(self): #, label_path = LABEL_PATH, train_path = TRAIN_PATH):
        self.label_list = []
        self.pi = {}
        self.trans = {}
        self.emit = {}
        self.word_set = set()
        # self.load_labels(label_path)


    def load_train(self, data_path):
        states = self.label_list
        f = open(data_path,"r", encoding="utf-8")
        lines = f.read().splitlines()

        word_list = []      # 训练集中所有字的集合
        global word_set
        count_dic = {}      # 记录每个字对应的状态出现的次数
        A_dic = {}      # 状态转移记录
        B_dic = {}      # 发射记录
        pi_dic = {}     # 初始状态记录
        # 初始化
        for s in states:
            count_dic[s] = 0
            A_dic[s] = {}
            for s2 in states:
                A_dic[s][s2] = 0
            B_dic[s] = {}
            pi_dic[s] = 0

        sentence_num = 0    # 记录句子个数（便于统计 pi ）
        new_flag = True     # 标记下一行是新句子开头
        i = 0
        sentence_state = [] # 记录一句话中每个字的状态（便于计算状态转移矩阵 A
        for line in lines:
            # print(line)
            # if line in ['\n', '\r\n']:  # 空行，下一行为新句子开头

            # 每行由 <observation(字)>  <state> 组成
            str_list = line.split()
            if len(str_list) == 0:
                new_flag = True # 标记新行（句子）
                sentence_state = []
                i = 0
                continue
            # print(str_list)
            character = str_list[0] # 观测值：字
            state = str_list[1]     # 状态：BWES
            word_list.append(character)
            sentence_state.append(state)
            count_dic[state] += 1
            if not B_dic[state].__contains__(character):
                B_dic[state][character] = 1 #0.0
            else:
                B_dic[state][character] += 1

            if new_flag:    # 新句子开头
                sentence_num += 1
                pi_dic[state] += 1
                new_flag = False
            else:   # 非句子开头才能记录状态转移
                A_dic[sentence_state[i - 1]][sentence_state[i]] += 1
            i += 1

        # 语料库中的字集合 ？
        word_set = word_set | set(word_list)
        # print("word_set:", len(word_set))

        f.close()
        # print(lines)

        # 初始状态概率 pi_dic
        for state in pi_dic:
            # if pi_dic[state] == 0:
            #     pi_dic[state] = infinite
            # else:
            #     pi_dic[state] = math.log(pi_dic[state] / sentence_num)
            pi_dic[state] = pi_dic[state] / sentence_num

        # 状态转移概率 A_dic(trans)
        for state_i in A_dic:
            log_sum = math.log(count_dic[state_i])
            sum = 0
            for state_j in A_dic[state_i]:
                sum += A_dic[state_i][state_j]
            
            for state_j in A_dic[state_i]:
                # if A_dic[state_i][state_j] == 0:
                #     A_dic[state_i][state_j] = infinite
                # else:
                #     A_dic[state_i][state_j] = math.log(A_dic[state_i][state_j]) - log_sum
                A_dic[state_i][state_j] = A_dic[state_i][state_j] / sum  # 状态i转移到状态j的概率 P(j_t|i_t-1)

        # 发射概率 B_dic(emit)
        for state in B_dic:
            sum = 0
            log_sum = math.log(count_dic[state])
            for character in B_dic[state]:
                # if B_dic[state][character] == 0:
                #     B_dic[state][character] = infinite
                # else:
                #     B_dic[state][character] = math.log(B_dic[state][character]) - log_sum
                B_dic[state][character] = B_dic[state][character] / count_dic[state]
                sum += B_dic[state][character]
            # print(sum)

        self.pi = pi_dic
        self.trans = A_dic
        self.emit = B_dic

        return pi_dic, A_dic, B_dic
        


    def load_labels(self, label_path):
        ''' 加载state标签列表 '''
        f = open(label_path,"r", encoding="utf-8")
        label_list = []
        lines = f.read().splitlines()
        for line in lines:
            label_list.append(line)
        print("label_list:",label_list)
        self.label_list = label_list
        return label_list


    def save_txt(self, save_dir, prefix):
        f = open(os.path.join(save_dir,prefix+ "pi.txt"),"w")
        f.write(str(self.pi))
        f.close()
        f = open(os.path.join(save_dir,prefix+"trans.txt"),"w")
        f.write(str(self.trans))
        f.close()
        f = open(os.path.join(save_dir,prefix+"emit.txt"),"w")
        f.write(str(self.emit))
        f.close()
        f = open(os.path.join(save_dir,prefix+"labels.txt"),"w")
        f.write(str(self.label_list))
        f.close()

    def load_txt(self, save_dir, prefix):
        f = open(os.path.join(save_dir,prefix+"pi.txt"),"r")
        self.pi = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"trans.txt"),"r")
        self.trans = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"emit.txt"),"r")
        self.emit = eval(f.read())
        f.close()
        f = open(os.path.join(save_dir,prefix+"labels.txt"),"r")
        self.label_list = eval(f.read())
        f.close()
        
        for tag in self.emit.keys():
            for w in self.emit[tag].keys():
                self.word_set.add(w)
        # print("word size:", len(self.word_set))
            

    
    def predict(self, observe):

        # f_test = open("C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\test_out.txt")
        ''' 输出预测状态 '''
        pi = self.pi
        trans = self.trans
        emit = self.emit
        states = self.label_list
        word_set = self.word_set

        V = [{}]
        path = {}
        observe_len = len(observe)

        temp_sum = 0
        for s in states:
            if observe[0] not in word_set:
                    # print(observe[0])
                    if s == 'S':
                        emit[s][observe[0]] = 0
                    else:
                        emit[s][observe[0]] = infinite

            # V[0][s] = pi[s] + emit[s].get(observe[0],infinite)  # dict.get(index, default_value)
            V[0][s] = pi[s] * emit[s].get(observe[0], 0)    # dict.get(index, default_value)
            temp_sum += V[0][s]
            path[s] = s
        # 归一化
        for s in states:
            V[0][s] /= temp_sum
        '''如何处理测试集中出现、训练集中没有出现的字符？
            若上一个字也没有出现：E-0，else- 负无穷 (当前字状态E概率最大)
            若上一个字出现了: B-0, else-负无穷  (当前字状态B概率最大)
        '''
        for i in range(1, observe_len):
            V.append({})
            newPath = {}
            prob_sum = 0.0

            if observe[i] not in word_set:
                # if observe[i - 1] not in word_set:
                #     if state == 'S':
                #         pass
                # # 上一个字已知 E S  BE SS
                # else:
                # print("fill:",observe[i])
                emit['S'][observe[i]] = 0.1   #0   
                emit['E'][observe[i]] = 0.5   #infinite
                emit['B'][observe[i]] = 0.4  #infinite
                emit['I'][observe[i]] = 0   #infinite

                # emit['S'][observe[i]] = infinite   
                # emit['E'][observe[i]] = 0   #infinite
                # emit['B'][observe[i]] = infinite
                # emit['I'][observe[i]] = infinite
                    
            for s2 in states:
                # (max_prob, max_s1) = max([ (V[i-1][s1] + trans[s1][s2], s1) for s1 in states])
                (max_prob, max_s1) = max([ (V[i-1][s1] * trans[s1][s2] , s1) for s1 in states])
                newPath[s2] = path[max_s1] + s2   # 拼接以状态 s2 结尾的新路径
                V[i][s2] = max_prob * emit[s2].get(observe[i],0)
                # V[i][s2] = max_prob + emit[s2].get(observe[i],infinite)

                prob_sum += V[i][s2]
            # 将每个O的状态概率和归一化，避免后面乘积太小溢出
            if prob_sum != 0:
                # print("observe:",observe[i])
                # print("V[i - 1]", V[i - 1])
                # print("V[i]:", V[i])
            # else:
                for s2 in states:
                    V[i][s2] /= prob_sum

            path = newPath
            # if observe[i] not in word_set:
            #     print(observe[i], V[i])

        (max_prob, max_s) = max( [ (V[observe_len - 1][s], s) for s in states] )
        # print("max_s:", max_s, observe[observe_len - 1])
        str = ""
        i = 0
        # for o, s in zip(observe, path[max_s]):
        #     # str += o + s + " " + str(V[i])
        #     print(o , s , " " ,V[i])
        #     i += 1
        return path[max_s]


    def print_emit(self, o):
        print("S",self.emit['S'].get(o,0))
        print("B",self.emit['B'].get(o,0))
        print("E",self.emit['E'].get(o,0))
        print("I",self.emit['I'].get(o,0))

        # 验证预测值，输出：正确个数/总个数
    def evaluate(self, sentence_list):
        correct = 0
        total = 0   # 总字数
        predict_list = []
        for sentence in sentence_list:
            if(len(sentence) == 0):
                continue
            total += len(sentence)
            input, expect = zip(*sentence)
            predict = self.predict(input)
            predict_list.append(predict)
            for i, (p, e) in enumerate(zip(predict, expect)):
                if p == e:
                    correct += 1
        return correct, total, predict_list
   
    # 读取样本
    def read_sentences(self, path):
        '''从训练集中读取sentence列表，每个sentence中为（字，state) tuples'''

        states = self.label_list
        f = open(path,"r", encoding="utf-8")
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


def main():
    # print(zip(list("123"),[1,2,3]))
    HMM = HmmModel()
    HMM.load_labels(LABEL_PATH)
    HMM.load_train(TRAIN_PATH)
    # HMM.save_model()
    HMM.save_txt(save_dir=".\\ModelSave\\hmm_model",prefix="12_total_")
    
    TEST_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12_eval.utf8"
    test_data = HMM.read_sentences(TEST_PATH)
    test_correct, test_total,_ = HMM.evaluate(test_data)
    test_accu =  test_correct/test_total * 100
    print("正确率：",test_accu,"%")

    # test_str = "我爱北京天安门"
    # print("测试用例：", test_str)
    # predict = HMM.predict(test_str)
    # for o, s in zip(test_str,predict):
    #     print(o,": " ,s)


def load_main():
    HMM = HmmModel()
    HMM.load_txt(save_dir=".\\ModelSave\\jxw_model")
    # HMM.save_txt(save_dir=".\\ModelSave\\hmm_model")
    test_str = "我爱北京天安门"
    print("测试用例：", test_str)
    predict = HMM.predict(test_str)
    for o, s in zip(test_str,predict):
        print(o,": " ,s)


# main()
# load_main()