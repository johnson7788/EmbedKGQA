import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class DatasetMetaQA(Dataset):
    def __init__(self, data, word2ix, relations, entities, entity2idx):
        self.data = data  # list, 208970条，训练数据数量， 每个数据的包含3个item，头实体，问句，答案实体列表
        self.relations = relations # 关系的嵌入： list，长度18，代表关系的数量，每个关系的的维度是400
        self.entities = entities   #实体的嵌入： list: 长度43234是实体的数量, 每个item是一个dict, 每个item是实体的对应的向量，每个实体的向量维度是400
        self.word_to_ix = {}   #单词到id的映射， 117个，  dict
        self.entity2idx = entity2idx  #实体到id的映射, dict , 43234个
        self.word_to_ix = word2ix
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())   # 保存每个实体


    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        """
        实体进行one_hot向量
        :param indices: eg: [17281]
        :type indices: list
        :return:
        :rtype:
        """
        indices = torch.LongTensor(indices) # 变成tensor, eg: tensor([37289])
        batch_size = len(indices)   # 获取数量个数
        vec_len = len(self.entity2idx)  #获取实体的总数量， eg: 43234
        one_hot = torch.FloatTensor(vec_len)  # 初始一个向量，维度43234
        one_hot.zero_()  # 全部变成0
        one_hot.scatter_(0, indices, 1)  # 只把索引的位置变成1
        return one_hot  # 维度是 43234， 43234是实体的数量

    def __getitem__(self, index):
        """
        获取每条数据
        :param index: 数据的索引, eg: 4586
        :type index: int
        :return:
        :rtype:
        """
        data_point = self.data[index]  #获取一条数据，数据的包含3个item，头实体，问句，答案实体列表, eg: ['The Marsh', 'NE directed_by', ['Jordan Barker']]
        question_text = data_point[1]   # 获取问题, eg: 'NE directed_by'
        question_ids = [self.word_to_ix[word] for word in question_text.split()] # 问题变成id，eg: [4, 99]
        head_id = self.entity2idx[data_point[0].strip()]  # 头实体变成id, eg: 33684
        tail_ids = []  # 存储尾实体变成id， eg: [17281], eg: [5683, 36879, 20347]
        for tail_name in data_point[2]:  # 处理尾实体，即答案
            tail_name = tail_name.strip()  # eg: 一个尾实体, 'Jordan Barker'
            tail_ids.append(self.entity2idx[tail_name])   # 尾实体变成id
        tail_onehot = self.toOneHot(tail_ids)  # 维度是 43234， 43234是实体的数量
        return question_ids, head_id, tail_onehot 


def _collate_fn(batch):
    """
    处理一个批次的数据
    :param batch: 每条数据的包含3个item，问题id， 头实体id，尾实体one_hot向量
    eg:
        0 = {list: 6} [0, 13, 12, 77, 15, 4]
        1 = {int} 6836
        2 = {Tensor: (43234,)} tensor([0., 0., 0.,  ..., 0., 0., 0.])
    :type batch: 一个批次的数据, list , eg: 1024
    :return:
    :rtype:
    """
    # 按问题的长度进行排序, 从长到短排序
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    # 问题的长度列表, 个数是batch_size, eg: 1024
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
    # 找出最长的样本, 即第一个样本
    longest_sample = sorted_seq_lengths[0]
    # minibatch_size ： 1024
    minibatch_size = len(batch)
    # print(minibatch_size)

    input_lengths = []
    p_head = []
    p_tail = []
    #初始化一个全0向量，维度是 [batch_size, seq_len]
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    for x in range(minibatch_size): # 遍历每个样本
        # 问题的变成id的列表， eg: [6, 30, 31, 19, 32, 33, 34, 0, 4, 18, 3]
        sample = sorted_seq[x][0]
        p_head.append(sorted_seq[x][1])  # 头实体变id, eg: p_head: [31021]
        tail_onehot = sorted_seq[x][2]  # 尾实体的one_host 向量, 维度 43234
        p_tail.append(tail_onehot)
        seq_len = len(sample)   #问题的长度
        input_lengths.append(seq_len)  # 长度列表
        sample = torch.tensor(sample, dtype=torch.long)  # 变成tensor格式
        sample = sample.view(sample.shape[0])
        inputs[x].narrow(0,0,seq_len).copy_(sample)  # 把向量拷贝到inputs中
    # 返回inputs： 问题的向量，维度是[batch_size, batch_max_seq_len],  问题的长度：[batch_size]， 问题中头实体的id, [batch_size],  答案尾实体的向量[batch_size, num_entities]
    return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(p_head, dtype=torch.long), torch.stack(p_tail)

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

    

