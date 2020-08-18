#-*- coding:utf-8 -*-
# author:allen
# datetime:2020-05-13 22:03
# software: PyCharm
# environment:
# packages:

import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import collections
import pandas as pd
import math




# 读取切分好的一行，返回词和词向量（numpy的矩阵）
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

#将glove.6B.100d.txt内的词读取并作为词典
with open('../data/glove.6B.100d.txt', 'r', encoding='utf-8') as emb_file:
    dict_length= 0
    emb_size = 100
    #print('dict_length: ', dict_length)
    #print('emb_size: ', emb_size)
    #dict_length, emb_size = int(dict_length), int(emb_size)
    # 对每一行做处理，结果存到顺序词典中
    emb = collections.OrderedDict(get_coefs(*l.rstrip().split()) for l in emb_file.readlines())
for k, v in emb.items():
    dict_length = dict_length+1
print("dict_length:",dict_length)
    #print(k, v.shape)
    #break


class Tokenizer:
    # 初始化的时候读取词表
    def __init__(self, vocab_list):
        self.vocab = self.load_vocab(vocab_list)
        for i, (k, v) in enumerate(self.vocab.items()):
            if i > 9:
                break
            print(k, v)

    # 读取词表
    def load_vocab(self, vocab_list):
        # 我们一般使用顺序字典来存储词表，这样能够保证历遍时index升序排列
        vocab = collections.OrderedDict()
        # 一般我们使用'UNK'来表示词表中不存在的词，放在0号index上
        vocab['UNK'] = 0
        index = 1
        # 依次插入词

        for token in vocab_list:
            token = token.strip()
            vocab[token] = index
            index += 1
        return vocab

    # 将单个字/词转换为数字id
    def token_to_id(self, token):
        # 不在词表里的词
        if token not in self.vocab.keys():
            return self.vocab['UNK']
        else:
            return self.vocab[token]

    # 将多个字/词转换为数字id
    def tokens_to_ids(self, tokens):
        ids_list = list(map(self.token_to_id, tokens))
        return ids_list

tokenizer = Tokenizer(emb.keys())


# 生成一个全0矩阵，大小为（词典长度+1，嵌入维度）
emb_matrix = np.zeros((1 + dict_length, emb_size), dtype='float32')

for word, id in tokenizer.vocab.items():
    emb_vector = emb.get(word)
    if emb_vector is not None:
        # 将编号为id的词的词向量放在id行上
        emb_matrix[id] = emb_vector
print(emb_matrix)
print(emb_matrix.shape)

from torch import nn

#使用lstm训练模型
class LSTMClassifierNet(nn.Module):
    def __init__(self, seq_length, label_len, hidden_dims=None, bidirectional=False, num_layers=1):
        super(LSTMClassifierNet, self).__init__()
        self.seq_length = seq_length
        self.label_len = label_len
        # 控制是否使用双向LSTM
        self.bidirectional = bidirectional
        if num_layers == 1:
            self.lstm_dropout = 0.0
        else:
            self.lstm_dropout = 0.2
        self.fc_dropout = 0.1

        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix))
        self.emb_size = self.emb.embedding_dim
        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = self.emb_size

        # 循环神经网络，输入为(seq_len, batch, input_size)，(h_0, c_0), 如果没有给出h_0和c_0则默认为全零
        # 输出为(seq_len, batch, num_directions * hidden_size), (h_final, c_final)
        # 关于hidden_state和cell_state，可以理解为“短期记忆”和“长期记忆”
        self.lstm = nn.LSTM(self.emb_size, self.hidden_dims,
                            num_layers=num_layers, dropout=self.lstm_dropout,
                            bidirectional=self.bidirectional)

        # 输出层，输入为(batch_size, hidden_dims)，输出为(batch_size, label_len)
        self.FC_out = nn.Sequential(
            nn.Linear(self.hidden_dims, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(self.fc_dropout),
            nn.Linear(50, self.label_len)
        )

        # softmax分类层
        self.softmax = nn.Softmax(dim=-1)
        # 交叉熵损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 通过词嵌入得到词的分布式表示，输出是(batch_size, seq_len, input_size)
        x = self.emb(x)
        # 但是LSTM要的输入是(seq_len, batch_size, input_size)，做一下维度变换
        # 你也可以在建立LSTM网络的时候设置"batch_first = True"，使得LSTM要的输入就是(batch_size, seq_len, input_size)
        x = x.permute(1, 0, 2)
        # 使用LSTM，输出为(seq_len, batch_size, num_directions * hidden_size)
        # LSTM输出的其实是最后一层的每个时刻的“短期记忆”
        x, (final_h, final_c) = self.lstm(x)
        # 我们就用最终的“长期记忆”来做分类，也就是final_c，它的维度是: (num_layers * num_directions, batch_size, hidden_size)
        # 我们把batch_size放到最前面，所以现在是(batch_size, num_layers * num_directions, hidden_size)
        final_c = final_c.permute(1, 0, 2)

        # 把每一层和每个方向的取个平均值，变成(batch_size, hidden_size)，现在就可以去做FC操作了
        final_c = final_c.sum(dim=1)

        logits = self.FC_out(final_c)
        if y is None:
            return logits
        else:
            return self.loss_fct(logits, y)


# 作文seq_length最长为800
seq_length = 800
# 得分从0到12分，有13类

label_len = 13
model = LSTMClassifierNet(seq_length, label_len, bidirectional=True)
# 使用print可以打印出网络的结构
print(model)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(str(total_trainable_params), 'parameters is trainable.')

if torch.cuda.is_available():
    model.to(torch.device('cuda'))



# 原始数据和标签
class data_example:
    def __init__(self, text, label):
        self.text = text
        self.label = label

# 处理完毕的数据和标签
class data_feature:
    def __init__(self, ids, label_id):
    #def __init__(self, ids, sents, words, avg_word_len, label_id):
        self.ids = ids
        #self.sents =sents
        #self.words = words
        #self.avg_word_len = avg_word_len
        self.label = label_id


# 读原始数据
examples = []
data = pd.read_excel('../data/作文打分语料（英语）.xlsx',sheet_name='训练语料')
essays = data["作文"]
scores = data["分数"]


# 读原始数据并全部转为小写字母

for i in range(len(essays)):
    examples.append(data_example(essays[i].lower(), scores[i]))

print('num of example: %d' % len(examples))
for i in range(3):
    print(examples[i].text, examples[i].label)

import nltk


def setence_count(essay):
    """
    文章中句子的个数
    """
    res = []
    i = 0
    for e in essay:
        if i<=4:
            print(nltk.tokenize.sent_tokenize(e))
            i+=1
        res.append(len(nltk.tokenize.sent_tokenize(e)))
    return res


setences_count = setence_count(essays)
for i in range(3):
    print("essay",i,"共有句子：",setences_count[i])


def word_count(essay):
    """
    文章中单词的总数
    """
    res = []

    for article in essay:
        count = 0
        words = nltk.word_tokenize(article)
        #print(words)
        #for ele in words:
            #count+=len(ele)
        count = len(words)
        res.append(count)
    return res


words_count = word_count(essays)
for i in range(3):
    print( "essay",i,"共有单词数：",words_count[i])


def avg_word_len(essay):
    """
    文章中单词的平均长度
    """
    res = []
    for i in range(len(essay)):
        tem = 1.0 * len(essay[i]) / words_count[i]
        res.append(tem)
    return res

avg_word_length = avg_word_len(essays)
for i in range(3):
    print( "essay",i,"单词平均长度：",avg_word_length[i])


def count_spell_error(essay):
    """
    文章中有多少个单词的拼错的（本程序认为不在glove.6B.100d.txt的词是错词）
    """
    res = []
    for article in essay:
        count = 0
        words = nltk.word_tokenize(article)
        #print(words)
        #for ele in words:
            #count+=len(ele)
        for word in words:
            if word not in emb.keys():
                count += 1
        res.append(count)
    return res

spell_errors = count_spell_error(essays)
for i in range(3):
    print( "essay",i,"spell_error数：",spell_errors[i])






# 处理原始数据
def convert_example_to_feature(examples,setences_count, words_count, avg_word_length):
    features = []
    j = 0
    for i in examples:
        # 使用tokenizer将字符串转换为数字id
        ids = tokenizer.tokens_to_ids(i.text)
        # 我们规定了最大长度，超过了就切断，不足就补齐（一般补unk，也就是这里的[0]，也有特殊补位符[PAD]之类的）
        if len(ids) > seq_length:
            ids = ids[: seq_length]
        else:
            ids = ids + [0] * (seq_length - len(ids))
        # 如果这个字符串全都不能识别，那就放弃掉
        if sum(ids) == 0:
            continue
        assert len(ids) == seq_length

        #将setences_count + words_count + avg_word_length,spell_errors添加进feature
        #因为最后要转为torch.long，所以avg_word_length[j]*10等于保留一位小数
        ids += [setences_count[j]] + [words_count[j]]  + [avg_word_length[j]*10] + [spell_errors[j]]
        j+=1

        features.append(data_feature(ids, i.label))
    return features




features = convert_example_to_feature(examples,setences_count, words_count, avg_word_length)

for i in range(3):
    print(features[i].ids, features[i].label)
    print(type(features[i].ids))

#将数据划分为80%训练集，20%验证集
from sklearn.model_selection import train_test_split
train, test = train_test_split(features, test_size = 0.2)


i = 0
for f in train:
    print("train.ids:",f.ids,"train.label:",f.label)
    i+=1
    if i == 3:
        break
i = 0
for f in test:
    print("test.ids:",f.ids,"test.label:",f.label)
    i+=1
    if i == 3:
        break





from torch.utils.data import TensorDataset, DataLoader

ids = torch.tensor([f.ids for f in train], dtype=torch.long)
label = torch.tensor([f.label for f in train], dtype=torch.long)

#载入数据
dataset = TensorDataset(ids, label)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

from torch.optim import Adam
# 1e-3 ~ 1e-5
optimizer = Adam(model.parameters(), lr=0.0025 )
print(optimizer)

#进行训练
epoch = 9
for i in range(epoch):
    total_loss = []
    for ids, label in dataloader:
        # 模型在GPU上的话
        if torch.cuda.is_available():
            ids = ids.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))
        # 因为我们这次loss已经写在模型里面了，所以就不用再计算模型了
        optimizer.zero_grad()
        loss = model(ids, label)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print("epoch: %d, loss: %.6f" % (i + 1, sum(total_loss) / len(total_loss)))



model.eval()



test_predict_tensor = model(torch.tensor([f.ids for f in test], dtype=torch.long))
print(test_predict_tensor.shape)

#找到最大可能的得分作为预测得分
def tensor_to_label(logits):
    logits = logits.detach().cpu().numpy()
    logits = np.argmax(logits, axis=-1)
    return logits.tolist()

test_predict = tensor_to_label(test_predict_tensor)


#打印真实得分和预测得分
print("test集真实得分：")
test_labels = []
for f in test:
    test_labels.append(f.label)
print(test_labels)
print("test集预测得分：")
print(test_predict)



from sklearn import metrics

#计算真实得分和预测得分的MSE
MSE = metrics.mean_squared_error(test_labels,test_predict)
print("MSE: ",MSE)

#计算真实得分和预测得分的RMSE
RMSE = math.sqrt(MSE)
print("RMSE: ",RMSE)

#计算真实得分和预测得分的Cohen’s κ
Cohen = metrics.cohen_kappa_score(test_labels,test_predict)
print("Cohen’s κ: ",Cohen)


test_predict_pd = pd.Series(test_predict)
test_labels_pd = pd.Series(test_labels)

#计算真实得分和预测得分的Pearson r
pearsonr = test_labels_pd.corr(test_predict_pd,method='pearson')
print("Pearson r: ",pearsonr)

#计算真实得分和预测得分的Spearman’s ρ
Spearman = test_labels_pd.corr(test_predict_pd,method='spearman')
print("Spearman’s ρ: ",Spearman)






