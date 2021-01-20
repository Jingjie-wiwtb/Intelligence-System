import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# from BiLSTM import BiLSTM_CRF

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    for w in seq:
        if w not in to_ix:
            to_ix[w] = len(to_ix)
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_batch(data ,word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad=[]
    tags_pad=[]
    for seq,tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len-len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len-len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    for seq in seqs_pad:
        for w in seq:
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
    # idxs = [to_ix[w] for w in seq_pad]
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_add(args):
    return torch.log(torch.sum(torch.exp(args), axis=0))



class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size + 100, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # 确保不会训成 pad
        # self.transitions.data[tag_to_ix[PAD_TAG], :] = -10000
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        begin = time.time()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to('cuda')
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # print('time consuming of crf_partion_function_prepare:%f' % (time.time() - begin))
        begin = time.time()
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = (forward_var + trans_score + emit_score)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        # print('time consuming of crf_partion_function1:%f' % (time.time() - begin))
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print('time consuming of crf_partion_function2:%f' %(time.time()-begin))
        return alpha

    def _forward_alg_new(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.).to('cuda')
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)  # +1
            aa = gamar_r_l + t_r1_k + self.transitions
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.)#.to('cuda')
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha


    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(embeds)
        #lstm_out = lstm_out.view(embeds.shape[1], self.hidden_dim)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        # score = autograd.Variable(torch.Tensor([0])).to('cuda')
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags.view(-1)])

        # if len(tags)<2:
        #     print(tags)
        #     sys.exit(0)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        #feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0])#.to('cuda')
        tags = torch.cat([torch.full([tags.shape[0],1],self.tag_to_ix[START_TAG]).long(),tags],dim=1)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:,-1]]
        return score



    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var.to('cuda') + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        # assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_new(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)#.to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()

        if start != self.tag_to_ix[START_TAG]:
            print(start, best_path)

        # assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_new(feats)
        gold_score = self._score_sentence(feats, tags)[0]
        return forward_score - gold_score

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq



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


def reload_net(model_path, word_ix_path):
    f = open(word_ix_path,"r")
    word_to_ix = eval(f.read())
    f.close()

    model = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, word_to_ix


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    EMBEDDING_DIM = 300 # 50  # 主要影响正确率
    print("embedding_dim:",EMBEDDING_DIM)
    HIDDEN_DIM = 128    #64    # 影响收敛速度   # 64 比128正确率好诶！
    print("hidden_dim",HIDDEN_DIM)

    # Make up some training data
    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]

    TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_10w.utf8"
    # TRAIN_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_total.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_600.utf8"
    # EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train1_10w.utf8"
    EVAL_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\dataset\\debug\\train2_1w.utf8"
    SAVE_FIG_PATH = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\lstm_figure"
    LOAD_DIR = "C:\\Users\\11752\\Desktop\\智能系统\\LAB2\\lab2_submission\\wordseg\\ModelSave\\BiLSTM"

    training_data = read_sentences(TRAIN_PATH)
    print("train_path:",TRAIN_PATH)
    eval_data = read_sentences(EVAL_PATH)
    print("eval_path:",EVAL_PATH)

    # print(training_data)

    word_to_ix = {}
    word_to_ix['<PAD>'] = 0
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "E": 2, "S":3,  START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
    ix_to_tag = dict([(v,k) for (k,v) in tag_to_ix.items()])


    model = BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.2, weight_decay=1e-4)

    reload_flag = False
    
    ######  加载模型训练  ######

    # reload_flag = True
    # model_path = os.path.join(LOAD_DIR, "train2_600-train2_600-50-64[0.37]")
    # word_ix_path = os.path.join(LOAD_DIR, "train2_600-train2_600-50-64word_to_ix.txt")
    # model, word_to_ix = reload_net(model_path,word_ix_path)
    # # print("word_to_ix",word_to_ix)

    #############################

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        _, predict_ix = model(precheck_sent)
        predict = [ix_to_tag[ix] for ix in predict_ix]
        print("predict",predict)
        # print(model(precheck_sent))
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    
    epoch_num = 20
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

    f = open(os.path.join(LOAD_DIR,model_prefix+"word_to_ix.txt"),"w")
    f.write(str(word_to_ix))
    f.close()

    for epoch in range(epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        # 清除梯度
        model.zero_grad()
        # 2. Get our batch inputs ready for the network,(turn them into Tensors of word indices.
        # If training_data can't be included in one batch, you need to sample them to build a batch
        sentence_in_pad, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
        # 前向传播计算 loss
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
        train_loss_list.append(loss)
        # 反向传导，梯度更新
        loss.backward()
        optimizer.step()

        # 检查验证集
        if epoch % 1 == 0:
            with torch.no_grad():
                sentence_in_pad, targets_pad = prepare_sequence_batch(eval_data, word_to_ix, tag_to_ix)
                eval_loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
                eval_loss_list.append(eval_loss)

                eval_correct = 0
                eval_size = 0
                # print(model(eval_data))
                for sentence in eval_data:
                    inputs, expects = sentence
                    eval_size += len(inputs)
                    # print("eval_inputs",inputs)
                    _, outputs_ix = model(prepare_sequence(inputs, word_to_ix))
                    outputs = [ix_to_tag[ix] for ix in outputs_ix]
                    eval_correct += sum(np.array(outputs) == np.array(expects))
                
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
                
                print("{}/{}, eval_accu:{:.6f}".format(epoch, epoch_num, eval_accu))

                # Check predictions before training
                with torch.no_grad():
                    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
                    _, predict_ix = model(precheck_sent)
                    predict = [ix_to_tag[ix] for ix in predict_ix]
                    print("predict",predict)

        

    # 保存模型数据
    best_eval_accu = best_eval_accu# .item()
    print(type(best_eval_accu))
    print("训练结束, best eval accu: ", best_eval_accu)

    log_path = os.path.join(LOAD_DIR,model_prefix+"["+"{:.2f}".format(best_eval_accu)+"]")
    print("保存最佳模型...")
    print("验证集accuracy：", best_eval_accu)
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
    plt.show()
    

