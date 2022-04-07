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
        tail_ids = []  # 存储尾实体变成id， eg: [17281]
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
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
    longest_sample = sorted_seq_lengths[0]
    minibatch_size = len(batch)
    # print(minibatch_size)
    # aditay
    input_lengths = []
    p_head = []
    p_tail = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    for x in range(minibatch_size):
        # data_a = x[0]
        sample = sorted_seq[x][0]
        p_head.append(sorted_seq[x][1])
        tail_onehot = sorted_seq[x][2]
        p_tail.append(tail_onehot)
        seq_len = len(sample)
        input_lengths.append(seq_len)
        sample = torch.tensor(sample, dtype=torch.long)
        sample = sample.view(sample.shape[0])
        inputs[x].narrow(0,0,seq_len).copy_(sample)

    return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(p_head, dtype=torch.long), torch.stack(p_tail)

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

    

