import torch
import numpy as np
import datetime
import torch.nn as nn
import torch.optim as optim
from BiLSTM import BiLSTM_CRF,BiLSTM_CRF2 
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LOAD_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\BiLSTM"
SAVE_FIG_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\lstm_figure"

def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    # return torch.tensor(idxs, dtype=torch.long)
    for w in seq:
        if w not in to_ix:
            # print("before:", len(to_ix))
            to_ix[w] = len(to_ix)
            # print("add ",w)
            # print("after:", len(to_ix))

    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def read_sentences(path):
    '''从训练集中读取sentence列表，每个sentence中为（[字]，[state[]) tuples'''

    f = open(path,"r", encoding="utf-8")
    lines = f.read().splitlines()
    sentence_list = []
    word_list = []
    sentence_state = []    
    for line in lines:
        # 每行由 <observation(字)>  <state> 组成
        str_list = line.split()
        if len(str_list) == 0:
            # new_flag = True# 标记新行
            sentence_list.append((word_list, sentence_state))
            word_list = []
            sentence_state = []
            continue
        # print(str_list)
        character = str_list[0] # 观测值：字s
        state = str_list[1]     # 状态：BWES
        word_list.append(character)
        sentence_state.append(state)

    print("sentence_size:",len(sentence_list))
    return sentence_list

'''从 data_path 重新加载CNN网络，返回实例'''
# def reload_net(data_path):
#     model = BiLSTM_CRF()
#     # checkpoint = torch.load(data_path)
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     return 

def reload_net(model_path, word_ix_path):
    # f = open(word_ix_path,"r",encoding='utf-8')
    # word_to_ix = eval(f.read())
    # f.close()
    f = open(word_ix_path,"r")
    word_to_ix = eval(f.read())
    print("word_to_ix size:",len(word_to_ix) )
    f.close()
        
    model = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, word_to_ix

def evaluate(model_path, word_ix_path,eval_data):
    eval_data = read_sentences(eval_path)
    print("eval_path:",eval_path)
    f = open(word_ix_path,"r",encoding='gb2312')
    word_to_ix = eval(f.read())
    f.close()
    model = torch.load(model_path)
    eval_correct = 0
    eval_size = 0
    # print(model(eval_data))
    for sentence, tags in eval_data:
        if len(sentence) == 0:
            continue;
        eval_size += len(sentence)                    
        sentence_in = prepare_sequence(sentence, word_to_ix)
        expects_ix = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        _, outputs_ix = model(sentence_in)
        # outputs = [ix_to_tag[ix] for ix in outputs_ix]
        eval_correct += sum(np.array(outputs_ix) == np.array(expects_ix))
        print(sentence)
        print(outputs_ix)
    eval_accu = eval_correct / eval_size
    print("accu:", eval_accu)

# C:\Users\11752\Desktop\智能系统\LAB2\lab2_submission\wordseg\ModelSave\BiLSTM\train12-train12_eval-15-32word_to_ix.txt
word_ix_path = os.path.join(LOAD_DIR, "train12-train12_eval-15-32word_to_ix.txt")
model_path = os.path.join(LOAD_DIR, "train12-train12_eval-15-32[0.92]")
eval_path = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_500.utf8"

# evaluate(word_ix_path, model_path,eval_path)        

def main(epoch_num, reload_flag = False):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    EMBEDDING_DIM = 15   #300 估计10000个字典，2^14
    print("embedding_dim:",EMBEDDING_DIM)
    HIDDEN_DIM = 32  #256
    print("hidden dim:", HIDDEN_DIM)

    # 小测试
    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]

    TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_total.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_3w.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_eval.utf8"

    EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train12_eval.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_2000.utf8"

    training_data = read_sentences(TRAIN_PATH)
    print("train_path:",TRAIN_PATH)
    eval_data = read_sentences(EVAL_PATH)
    print("eval_path:",EVAL_PATH)

    word_to_ix = {}
    # word_to_ix['<PAD>'] = 0
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "E": 2, "S":3,  START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
    ix_to_tag = dict([(v,k) for (k,v) in tag_to_ix.items()])

    print("word_to_ix", len(word_to_ix))

    if reload_flag == False:
        model = BiLSTM_CRF2(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    
    ######  加载模型训练  ######
    if reload_flag == True:
        # model_path = os.path.join(LOAD_DIR, "train2_600-train2_600-50-64[0.37]")
        # word_ix_path = os.path.join(LOAD_DIR, "train2_600-train2_600-50-64word_to_ix.txt")
        # C:\Users\11752\Desktop\智能系统\LAB2\lab2_submission\wordseg\ModelSave\BiLSTM\train2_3w-train1_2000-15-8[0.79]
        # word_ix_path = os.path.join(LOAD_DIR, "train2_3w-train1_2000-15-8word_to_ix.txt")
        word_ix_path = os.path.join(LOAD_DIR, "train12-train12_eval-30-8test_word_to_ix.txt")
        # model_path = os.path.join(LOAD_DIR, "train1_total-train1_10w-14-4[0.87]")
        # model_path = os.path.join(LOAD_DIR, "train2_3w-train1_2000-15-8[0.79]")
        model, word_to_ix = reload_net(model_path,word_ix_path)
        print("reloaded!")
        # print("word_to_ix",word_to_ix)
        ##  重新保存模型
        state = {'state_dict':model.state_dict()}
        torch.save(state, os.path.join(LOAD_DIR, "state_dict.lstm"))
        print("state_dict saved!")

    #############################
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        # print(model(precheck_sent))
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    
    best_eval_accu = 0.0
    best_model = None
    eval_accu_list = []
    train_file = os.path.basename(TRAIN_PATH)[0:-5]
    eval_file = os.path.basename(EVAL_PATH)[0:-5]
    model_prefix = train_file + "-" + eval_file + "-"+str(EMBEDDING_DIM) + "-" + str(HIDDEN_DIM)
    plt.figure(model_prefix)
    # train_accu_list = []

    eval_accu_list = []
    eval_loss_list = []
    train_loss_list = []

    
    f = open(os.path.join(LOAD_DIR,model_prefix+"test_word_to_ix.txt"),"w")
    print("word_to_ix saved in ", model_prefix)
    f.write(str(word_to_ix))
    f.close()

    for epoch in range(epoch_num): 
        for sentence, tags in training_data:
            if len(sentence) == 0:
                continue;
            # 清除梯度
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

        # 检查验证集
        if epoch % 1 == 0:
            with torch.no_grad():
                eval_correct = 0
                eval_size = 0
                # print(model(eval_data))
                for sentence, tags in eval_data:
                    if len(sentence) == 0:
                        continue;
                    eval_size += len(sentence)                    
                    sentence_in = prepare_sequence(sentence, word_to_ix)
                    expects_ix = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
                    _, outputs_ix = model(sentence_in)
                    # outputs = [ix_to_tag[ix] for ix in outputs_ix]
                    eval_correct += sum(np.array(outputs_ix) == np.array(expects_ix))
 
           
                # for sentence in eval_data:
                #     inputs, expects = sentence
                #     eval_size += len(inputs)
                #     # print("eval_inputs",inputs)
                #     _, outputs_ix = model(prepare_sequence(inputs, word_to_ix))
                #     outputs = [ix_to_tag[ix] for ix in outputs_ix]
                #     eval_correct += sum(np.array(outputs) == np.array(expects))
                
                eval_accu = eval_correct / eval_size
                eval_accu_list.append(eval_accu)
                # 保存accu_list
                f = open(os.path.join(LOAD_DIR,model_prefix+"eval_accu.txt"),"w")
                f.write(str(eval_accu_list))
                f.close()

                if eval_accu > best_eval_accu:
                    best_eval_accu = eval_accu
                    best_model = model
                    # best_cache["optimizer_state_dict"] = optimizer.state_dict
                    # 保存模型
                    print("保存模型...")
                    print("验证集accuracy：", best_eval_accu)
                    log_path = os.path.join(LOAD_DIR,model_prefix+"["+"{:.2f}".format(best_eval_accu)+"]")
                    torch.save(best_model, log_path)
                
                print("{}/{}, eval_accu:{:.6f}".format(epoch, epoch_num, eval_accu),"   time: ", datetime.datetime.now())

                # Check predictions before training
                with torch.no_grad():
                    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
                    _, predict_ix = model(precheck_sent)
                    predict = [ix_to_tag[ix] for ix in predict_ix]
                    # print("predict",predict)

    # Check predictions after training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #     print(model(precheck_sent))
        # We got it!

     # 保存模型数据
    print("训练结束, best eval accu: ", best_eval_accu)

    log_path = os.path.join(LOAD_DIR,model_prefix+"["+"{:.2f}".format(best_eval_accu)+"]")
    print("保存最佳模型...(accuracy：", best_eval_accu)
    torch.save(best_model, log_path)
    print("训练完成")

    # 画图
    plt.subplot(1,2,1)
    plt.title("accu")
    plt.ylim(0,1)
    
    plt.plot(eval_accu_list, label =  "eval_accu_list")
    plt.subplot(1,2,2)
    plt.title("loss")
    plt.plot(eval_loss_list,  label = "eval_loss_list")
    plt.plot(train_loss_list,  label = "train_loss_list")

    plt.legend()
    plt.savefig(os.path.join(SAVE_FIG_PATH, model_prefix +".jpg"))
    print(" Figure saved in ", SAVE_FIG_PATH, model_prefix)
        
    plt.show()

main(20, True)
# main(1)