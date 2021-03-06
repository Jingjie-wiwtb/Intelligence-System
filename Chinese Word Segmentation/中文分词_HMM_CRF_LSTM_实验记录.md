# **Lab 2  实验报告**

**贺劲洁  18307130370**

[TOC]



### **一、HMM**

##### 【对溢出的处理】

概率值累乘次数太多时会变成0

两种解决：1. 对所有概率值进行log处理；  2. 每次对 Viterbi 结果进行归一化处理

对比：（训练集：250w, 验证集：10w)

| 解决方案 | 正确率   |
| -------- | -------- |
| log 处理 | 81.711 % |
| 归一化   | 82.678 % |



##### 【验证集对正确率的影响】

一开始验证集划得很小，后来发现这对正确率的评估影响很大



##### 【实验——对生字的处理】

推测生字为"B","E"状态的概率更大，因此对生字的发射矩阵赋值

| 生字的发射矩阵赋值   | 正确率 |
| -------------------- | ------ |
| B:0.5, E:0.5, else:0 | 82.67  |
| E:1,  else:0         | 81.71  |
| B: 0.4, E:0.5, S:0.1 | 82.85  |

思考：

1. 对生字标签的预测也会影响后面的预测，对正确率的影响较大。实验方案中 `B: 0.4, E:0.5, S:0.1` 的效果最好
2. 对生字分词的判断还与前一个字有关，若上一个字也没有出现，当前字字状态E概率最大； 否则当前字状态B概率最大。但是生字出现的概率本身较小，不具有统计学意义，没有进一步实验。



### **二、CRF**

论文中将参数初始化为0，再通过预测逐步调整，而本模型在读取训练集时，同时类似HMM做了状态转移概率、发射概率、初始状态做出了统计，将参数置为相应出现次数，从而增加了收敛速度，能够较快达到较为理想的收敛效果。

#### **实验**

【模板对正确率的影响】

##### 【实验一】

```python
公共模板：
# Unigram
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U05:%x[-2,0]/%x[-1,0]
U06:%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]
U09:%x[-1,1]/%x[0,1]
```

**变量1** 窗口大小

```
%x[-2,0]
%x[2,0]
```

**变量二** 前一个字和当前字的标签转移

```
%x[-1,1]/%x[0,1]
```



![image-20201219175615435](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201219175615435.png)

1. 两种情况下，不使用模板`U[-1,1]/[0,1]`， 即不使用前后两个字的状态转移时，收敛速度更快、正确率相近
2. 窗口为5（即前后各考虑两个字）的模板收敛到的正确率更高，收敛速度相近



##### 【实验二  单字state模板实验】

```python
# 公共模板：
# Unigram
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U05:%x[-2,0]/%x[-1,0]
U06:%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]
U09:%x[-1,1]/%x[0,1]
```

**变量：**

```
无 %x[-1,0]
无 %x[0,0]
无 %x[1,0]

无 %x[-1,0]
   %x[0,0]
   %x[1,0]
   
(不考虑两个字的状态转移)无:
U05:%x[-2,0]/%x[-1,0]
U06:%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]
```

![image-20201219182617975](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201219182617975.png)

**分析**：没有`[0,0]`的影响最大, 没有`[-1,0]` 的影响最小



##### 【实验三 双字state模板实验】

```python
# 公共模板：
# Unigram
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U06:%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]
U09:%x[-1,1]/%x[0,1]
```

**变量**

```
no_6  %x[-1,0]/%x[0,0]
no_7  %x[-1,0]/%x[1,0]
no_8  %x[0,0]/%x[1,0]
```

![image-20201220121313298](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201220121313298.png)

**分析**	观察可见， 缺少模板 `%x[0,0]/%x[1,0]` 对收敛正确率的影响最大， 但本次实验中，去掉模板 7 `%x[-1,0]/%x[1,0]` 的效果似乎更好



##### 【实验四 Bigram 的使用】

根据上述实验结果尝试是否有更好的模板

```python
# 公共模板：
# Unigram
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U05:%x[-1,0]/%x[0,0]
U06:%x[0,0]/%x[1,0]
# U07:%x[-1,0]/%x[1,0]
U08:%x[0,1]
U09:%x[-1,1]/%x[0,1]

# Bigram
```

**变量：**

```
win5_no8:%x[0,1]
win5_no7  %x[-1,0]/%x[1,0]
```

![image-20201220141152786](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201220141152786.png)

**分析**	对比发现，去掉模板 ` %x[-1,0]/%x[1,0]` 的收敛效果更好，而去掉模板 `%x[0,1]` 与使用该模板的收敛曲线几乎完全重合，说明该模板对分词的预测作用不大



【实验五 Bigram效果】

```
# Unigram
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U06:%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]
U09:%x[-1,1]/%x[0,1]

# Bigram
```

变量：

```
# Bigram
1. B01:%x[-1,0]
2. B02:%x[0,0]
3. B03:%x[1,0]
```

![image-20201222092937511](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201222092937511.png)

观察发现，在本实验任务中，增加Bigram模板对收敛效果增益不大，且会大大增加训练量，因此在最终的模板中没有使用Bigram类型模板



### **三、BiLSTM + CRF**

##### 【理解】

​	循环神经网络(RNN)的隐藏层的值s不仅仅取决于当前这次的输入x，还取决于上一次隐藏层的值s。权重矩阵 W就是隐藏层上一次的值作为这一次的输入的权重。但是，RNN在训练中很容易发生梯度爆炸和梯度消失，这导致训练时梯度不能在较长序列中一直传递下去，从而使RNN并不能很好的处理较长的序列。

长短时记忆网络(LSTM) 解决了原始循环神经网络的缺陷。它用遗忘门决定上一时刻的单元状态有多少保留到当前时刻；输入门决定当前时刻网络的输入有多少保存到单元状态。输出门来控制单元状态有多少输出到LSTM的当前输出值。

LSTMLayer的参数包括输入维度、输出维度、隐藏层维度，单元状态维度等于隐藏层维度。gate的激活函数为sigmoid函数，输出的激活函数为tanh。

同时，对于语言模型来说，很多时候光看前面的词是不够的，BiLSTM 用正反向的输出同时决定最终的输出。



##### 【对生字的处理】

​	初始化时将 lstm 层字典大小扩大（大于训练集的字典大小），验证发现生字时，将生字加入字典集。



【参数】

`num_embeddings`  – 词典的大小尺寸，

`embedding_dim`  语料中每一个单词对应的其相应的词向量

`hidden_size` 隐藏层节点个数

由于训练速度太慢，尝试过用batch，但是对内存要求太高

![image-20201220125039698](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201220125039698.png)



##### 【实验】embedding_dim 和 hidden_size 的选择

![image-20201220143349600](C:%5CUsers%5C11752%5CDocuments%5C%E6%88%91%E7%9A%84%E5%9D%9A%E6%9E%9C%E4%BA%91%5Cupload%5Cimage-20201220143349600.png)



相同词向量维度的条件下，hidden_dim 越大，收敛速度越快

相同 hidden_dim 的条件下，实验中的词向量维度越大，收敛正确率越高

综合考虑收敛速度和电脑性能，在正式训练时选择了参数 embedding_dim = 15, hidden_dim = 32



### 四、HMM、CRF、LSTM 的对比分析

##### 【实验】预测不对的字

1. 和数据集的标签标注有关，以 CRF 的训练结果为例，用 dataset2 进行训练，在 dataset1 中抽取验证集，发现：

dataset2 中形如 ”各地“的标签为 ”BE“， 而 dataset1 中 为 ”SS“， 因此用dataset2训练的模型在dataset1中总是预测错误

>   【字】【×】【√】
>
>   国   I   E
>   周   S   B
>   根   B   I
>   国   I   E
>
>   委   E   B
>   间   E   S
>
>   由   B   S
>   此   E   S
>   一   B   S
>   个   E   S
>   年   S   E
>   国   I   E
>   全   B   S
>   国   E   S
>   各   B   S
>   地   E   S
>   各   B   S
>   地   E   S
>   全   B   S
>   国   E   S
>   融   I   E



2. 和使用的训练方法有关

【实验】预测效果的对比分析

```
['我爱北京天安门', '今天天气怎么样']
正确：
['SSBEBIE', 'BEBEBIE']

【HMM】
['BEBESBE', 'BEBEBIE']

【Crf】
['SSBEBIE', 'BEBEBIB']

【BiLSTM+CRF】
['BEBESBE', 'BESSBES']
```

​	观察发现，“我爱北京天安门“这句话，只有CRF能够正确预测，HMM 和 BiLSTM 无法正确预测，猜测原因为 CRF 使用了模板，考虑了除了当前字和相邻字的标签，还能根据模板考虑其他字和标签之家你的关联；而HMM 和 BiLSTM 只能考虑相邻字之间的状态转移，因此在分词任务中，CRF 能够更好地根据前后语境对标签进行预测。



#### 对 HMM、CRF、LSTM 的对比思考

**【HMM】**

​	HMM模型基于两个假设：输出观察值之间严格独立；状态的转移过程中当前状态只与前一状态有关。然而实际上，序列标注问题不仅和单个词相关，还和观察序列的长度，单词的上下文等因素相关。

**【CRF】**

​	和 HMM相比， CRF能够自定义特征函数，从而考虑上下文特征。

​	和 LSTM相比，CRF虽然不能考虑长远的上下文信息，但能通过使LSTM一样将每个时刻的最优结果拼接起来。

**【LSTM】**

​	和 CRF 相比， LSTM作为RNN模型中的一类，除了能够捕捉到较长的上下文信息，还具备神经网络拟合的能力。输出层受到包含上下层信息的隐层和当前输入的共同影响，但和其他时刻的输出时独立的，因此当不同时刻的输出之间存在较强的依赖关系，例如“名词后接动词”等类似的特征难以建模。

​	综上所述，LSTM 和CRF 的效果根据任务场景的不同有所区别，应结合网络复杂度等综合因素进行选择、或根据各自的优势结合使用。









