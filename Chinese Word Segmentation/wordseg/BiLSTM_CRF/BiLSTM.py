import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

'''
    【log_sum_exp():https://blog.csdn.net/zziahgf/article/details/78489562】
    log∑exp{xn} = a + log∑exp{xn−a}
    a为任意数.
    上式意味着，我们可以任意的平移指数的中心值， 放大或缩小.
    一种典型的是使 a = max{xn}
    这样保证了取指数时的最大值为0,使得绝对不会出现上溢(overflow)，即使其余的下溢(underflow)，也可以得到一个合理的值.
'''
# def log_sum_exp(vec):   # vec.size() = [1, n]
#     max_score = vec[0, np.argmax(vec)]  # 每一行的最大值
#     max_score_broadcast = max_score.view(1,-1).expand(1, vec.size()[1]) # 生成(1*n)大小的最大值列向量
#     # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
#     return max_score + torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size    # 词典大小
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        ''' torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
            - num_embeddings (python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
            - embedding_dim (python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
        '''
        self.word_embeds = nn.Embedding(vocab_size + 100, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers = 1, bidirectional=True)#  "//":整除  batch_first:parallel
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))   # 随机初始化转移矩阵
        
        # 限制状态不会：从STOP开始 or 在START结束
        self.transitions.data[tag_to_ix[START_TAG],:] = -10000  # 其他状态到START_TAG的概率
        self.transitions.data[:,tag_to_ix[STOP_TAG]] = -10000 # STOP_TAG到其他状态的概率
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))


    # 从 lstm 获得 feats 输出
    # 维度=len(sentence)*tagset_size，表示句子中每个词是分别为target_size个tag的概率 
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 前向传播，根据当前参数预测所有路径的得分
    # log ∑exp{score(x)}
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        # 将 start 的值置为0， 表示开始进行网络的传播 ？
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0

        # warp in a variable so that we will get automatic backprop ？
        forward_var = init_alphas   # 初始状态的 forward_var, 随 step 变化

        # iterate through the sentence  ?
        for feat in feats:
            alphas_t = []   # 该 timestep 的钱箱tensor
            for next_tag in range(self.tagset_size):
                # 生成 1*tagset_size 的发射矩阵
                emit_score = feat[next_tag].view(1,-1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)    # 相当于把行向量改成列向量
               
                # edge(i -> next_tag):
                next_tag_var = forward_var + trans_score + emit_score

                # The forward variable for this tag is log-sum-exp of all the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 到第（t-1）step时５个标签的各自分数   ?
            forward_var = torch.cat(alphas_t).view(1, -1)
        
        # 最后只将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var) # alpha是一个0维的tensor

        return alpha


    '''
        _forward_a(self, feats) 用随机转移矩阵计算出的最大可能路径，
        _score_sentence(self, feats, tags): 用随机转移矩阵和真是标签计算出的 score
    '''
    # 根据真实标签计算出的 score
    # score() = ∑trans(i->i+1) + ∑emit(i)
    def _score_sentence(self, feats, tags):
        # score = torch.zeros(1)
        score = torch.zeros(tags.shape[0])

        # 将 START_TAG 的标签拼接到tag序列上
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # for i, feat in enumerate(feats):
        for i in range(feats.shape[1]):
            # self.transitions[tags[i + 1], tags[i]] 从标签i到标签i+1的转移概率
            score += self.transitions[tags[i+1], tags[i]] + feats[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score


    # 维特比解码，输出得分和路径
    def _viterbi_decode(self, feats):
        backpointers = []   # 记录路径

        # 初始化
        init_vvars = torch.full((1, self.tagset_size), -10000.) # 初始化为 log(0)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0    # log(1)

        # 保存前一层 Viterbi ，用变量表示可以节省空间，不需要存 Viterbi 矩阵
        forward_var = init_vvars
        # forward_var_list = []
        # forward_var_list.append(init_vvars)

        # V[i][s2] = max([ (V[i-1][s1] + trans[s1][s2]) + emit[s2]
        for feat in feats:
            bptrs_t = [] # 保存路径
            viterbivars_t = [] # 当前层 Viterbi, viterbivars_t[i] 表示标签i对应的概率
            
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i
                # at the previous step, plus the score of transitioning
                # from tag i to next_tag
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag] 
                best_tag_id = argmax(next_tag_var)  # 最好的标签
                bptrs_t.append(best_tag_id) # 记录路径
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # append(一个数），转置为行向量(1*tag_size)

            # + emit[]
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t) # bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 回溯最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 删除 START 标记
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] # Sanity check
        # 把从后向前的路径正过来
        best_path.reverse()
        return path_score, best_path

    # loss function
    def neg_log_likelihood(self, sentence, tags):
        # feats: 经过 LSTM + Linear 的输出，作为 CRF 的输入，相当于发射概率矩阵，表示每个word对应每个label的得分
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # 获取 BiLSTM 的输出得分
        lstm_feats = self._get_lstm_features(sentence)

        # 对 BiLSTM 输出 进行Viterbi预测
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



class BiLSTM_CRF2(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size+100, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)     # Maps the output of the LSTM into tag space

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))    #随机初始化转移矩阵

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000       #tag_to_ix[START_TAG]: 3（第三行，即其他状态到START_TAG的概率）
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000        #tag_to_ix[STOP_TAG]: 4（第四列，即STOP_TAG到其他状态的概率）
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    #所有路径的得分，CRF的分母
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)        #初始隐状态概率，第1个字是O1的实体标记是qi的概率
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas          #初始状态的forward_var，随着step t变化

        for feat in feats:                 #feat的维度是[1, target_size]
            alphas_t = []
            for next_tag in range(self.tagset_size):      #给定每一帧的发射分值，按照当前的CRF层参数算出所有可能序列的分值和

                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size) #发射概率[1, target_size] 隐状态到观测状态的概率
                trans_score = self.transitions[next_tag].view(1, -1)                #转移概率[1, target_size] 隐状态到下一个隐状态的概率
                next_tag_var = forward_var + trans_score + emit_score               #本身应该相乘求解的，因为用log计算，所以改为相加

                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]     #最后转到[STOP_TAG]，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(terminal_var)

    #得到feats，维度=len(sentence)*tagset_size，表示句子中每个词是分别为target_size个tag的概率
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # print(len(sentence))
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    #正确路径的分数，CRF的分子
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            #self.transitions[tags[i + 1], tags[i]] 是从标签i到标签i+1的转移概率
            #feat[tags[i+1]], feat是step i的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]     # 沿途累加每一帧的转移和发射
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]               # 加上到END_TAG的转移
        return score


    #解码，得到预测序列的得分，以及预测的序列
    def _viterbi_decode(self, feats):
        backpointers = []                #回溯路径；backpointers[i][j]=第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是什么状态

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):

                next_tag_var = forward_var + self.transitions[next_tag]            #其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)                                 #选择概率最大的一条的序号
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)            #从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)


        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]    #其他标签到STOP_TAG的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):           #从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]        #安全性检查
        best_path.reverse()                              #把从后向前的路径倒置
        return path_score, best_path

    #求负对数似然，作为loss
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)        #emission score
        forward_score = self._forward_alg(feats)         #所有路径的分数和，即b
        gold_score = self._score_sentence(feats, tags)   #正确路径的分数，即a
        return forward_score - gold_score                #注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)


    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq