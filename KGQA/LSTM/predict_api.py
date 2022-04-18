#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/18 10:49 上午
# @File  : predict.py.py
# @Author:
# @Desc  :  预测接口

import os

import self as self
import torch
import numpy as np
from tqdm import tqdm
import json
from model import RelationExtractor
from flask import Flask, request, jsonify, abort
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")
app = Flask(__name__)


class LSTMKQGA():
    def __init__(self, model_name="ComplEx", embedding_folder="../../pretrained_models/embeddings/ComplEx_MetaQA_full/"):
        """
        :param model_name: 使用的哪个模型
        """
        self.embedding_dim = 256
        self.hidden_dim = 200
        self.relation_dim = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "../../checkpoints/MetaQA/best_score_model.pt"
        train_data_path = '../../data/QA_data/MetaQA/qa_train_1hop_half.txt'
        entity_embedding_path = embedding_folder + '/E.npy'
        relation_embedding_path = embedding_folder + '/R.npy'
        entity_dict = embedding_folder + '/entities.dict'
        relation_dict = embedding_folder + '/relations.dict'
        w_matrix = embedding_folder + '/W.npy'
        bn_list = []

        for i in range(3):
            bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
            bn_list.append(bn.item())
        entities = np.load(entity_embedding_path)  # 实体嵌入【实体个数，嵌入维度】 ,(43234, 400)
        relations = np.load(relation_embedding_path)  # 关系嵌入 (18, 400)， 【关系种类，嵌入维度】
        # 返回e是实体对应的嵌入，字典格式， r是关系对应的嵌入，字典格式,
        e, r = self.preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
        # 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
        entity2idx, idx2entity, embedding_matrix = self.prepare_embeddings(e)
        self.entity2idx = entity2idx
        self.idx2entity = idx2entity
        # 处理数据，data， list，  是【头实体，问题，答案列表】的格式，如果split是FALSE., 这里是208970条训练数据
        data = self.process_text_file(train_data_path)
        # 对问题进行处理，获取单词到id， id到单词的映射字典，最大长度
        word2idx, idx2word, max_len = self.get_vocab(data)
        self.word2idx = word2idx
        # 初始化数据集
        # 初始化dataloader, batch_size: 1024
        # 初始化模型， embedding_dim： 256， hidden_dim： 200，vocab_size：117， 一共117个单词，问题的单词数量
        model = RelationExtractor(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, vocab_size=len(word2idx),
                                  num_entities=len(idx2entity), relation_dim=self.relation_dim,
                                  pretrained_embeddings=embedding_matrix, freeze=True, device=self.device, entdrop=0.0,
                                  reldrop=0.0, scoredrop=0.0, l3_reg=0.0, model=model_name, ls=0.0,
                                  w_matrix=w_matrix, bn_list=bn_list)
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        self.model = model
    def preprocess_entities_relations(self,entity_dict, relation_dict, entities, relations):
        """
        获取实体的嵌入向量和关系的嵌入向量的字典
        :param entity_dict: '../../pretrained_models/embeddings/ComplEx_MetaQA_full/entities.dict'
        :type entity_dict:
        :param relation_dict:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/relations.dict'
        :type relation_dict:
        :param entities: (43234, 400) 实体嵌入矩阵
        :type entities:
        :param relations:  (18, 400)  关系嵌入矩阵
        :type relations:
        :return:
        :rtype:
        """
        e = {}
        r = {}
        f = open(entity_dict, 'r')
        for line in f:  # 每条数据格式: line: '1\t$9.99\n'
            line = line.strip().split('\t')  # ['1', '$9.99']
            ent_id = int(line[0])  # 获取实体id， int
            ent_name = line[1]  # 获取实体名称, $9.99
            e[ent_name] = entities[ent_id]  # 获取实体的嵌入向量, 维度 400, 实体名称：实体向量的字典格式
        f.close()
        # e：43234个，代表实体的个数
        f = open(relation_dict, 'r')  # 打开关系的嵌入
        for line in f:
            line = line.strip().split('\t')  # ['0', 'directed_by']
            rel_id = int(line[0])  # int， 关系的id, eg: 0
            rel_name = line[1]  # 'directed_by'
            r[rel_name] = relations[rel_id]  # 这个关系对应向量
        f.close()  # 关系r, dict, 18个关系
        return e, r
    def get_vocab(self, data):
        """
        对问题进行处理，获取单词到id， id到单词的映射字典，最大长度
        :param data:
        :type data:
        :return:
        :rtype:
        """
        word_to_ix = {}
        maxLength = 0
        idx2word = {}
        for d in data:
            sent = d[1]
            for word in sent.split():
                if word not in word_to_ix:
                    idx2word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)

            length = len(sent.split())
            if length > maxLength:
                maxLength = length

        return word_to_ix, idx2word, maxLength
    def prepare_embeddings(self,embedding_dict):
        """
         生成 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
        :param embedding_dict:
        :type embedding_dict:
        :return: 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
        :rtype:
        """
        entity2idx = {}
        idx2entity = {}
        i = 0
        embedding_matrix = []
        for key, entity in embedding_dict.items():  # key代表每个实体, 例如：'$'，  entity代表每个实体的嵌入向量，400维度
            entity2idx[key.strip()] = i  # 实体到id的映射, eg: {'$': 0}
            idx2entity[i] = key.strip()  # id到实体的映射, eg: {0: '$'}
            i += 1
            embedding_matrix.append(entity)
        return entity2idx, idx2entity, embedding_matrix

    def process_text_file(self, text_file):
        """
        获取训练数据，【头实体，问题，答案列表】的格式，如果split是FALSE， 如果是True，变成【头实体，问题，答案]的格式
        :param text_file: '../../data/QA_data/MetaQA/qa_train_1hop.txt'
        :type text_file:
        :param split: 如果分割，那么每个答案变成由列表变成单个字符
        :type split:
        :return:
        :rtype:
        """
        data_file = open(text_file, 'r')
        data_array = []
        for data_line in data_file.readlines():  # 读取训练集的每一行
            data_line = data_line.strip()
            if data_line == '':
                continue
            data_line = data_line.strip().split('\t')  #
            question = data_line[0].split('[')
            question_1 = question[0]  # 问题的前半部分
            question_2 = question[1].split(']')  # 问题对应的实体
            head = question_2[0].strip()  # 问题对应的实体
            question_2 = question_2[1]  # 问题的后半部分
            question = question_1 + 'NE' + question_2  # 问题变成NE连接， 'what movies are about NE'
            ans = data_line[1].split('|')  # 答案变成列表格式:['Top Hat', 'Kitty Foyle', 'The Barkleys of Broadway']
            data_array.append([head, question.strip(), ans])  # 一条数据变成【头实体，问题，答案列表】的格式
        return data_array
    def predict_from_file(self, test_data_path='../../data/QA_data/MetaQA/qa_test_1hop.txt'):
        """
        使用文件进行预测
        :param test_data_path:
        :type test_data_path:
        :return:
        :rtype:
        """
        data = self.process_text_file(test_data_path)
        answers = self.predict(data)
        return answers

    def predict(self, data):
        """
        data: 数据只需要， 头实体+ 问题
        :param data:  每条数据都是[头实体，问题，尾实体列表] 的格式, eg: ['Grégoire Colin', 'what does NE appear in', ['Before the Rain']]
        :type data:
        :return:
        :rtype:
        """
        answers = []
        list_data = self.data_generator(data=data, word2ix=self.word2idx, entity2idx=self.entity2idx)
        # list_data： 返回的处理后的数据
        for d in tqdm(list_data, desc="模型预测中"):
            head = d[0].to(self.device)
            question = d[1].to(self.device)
            # 问题的长度
            ques_len = d[2].unsqueeze(0)
            top_2 = self.model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]

            answer_text = self.idx2entity[pred_ans]
            #问题
            question_text = d[-1]
            # 问题的头实体
            qestion_head = d[-2]
            # 替换 NE 为问题的头实体
            quesiton = question_text.replace(' NE ', f" {qestion_head} ")
            answers.append(f"问题是:{quesiton}, 答案是: {answer_text}")
        return answers

    def data_generator(self,data, word2ix, entity2idx):
        list_data = []
        for i in range(len(data)):
            data_sample = data[i]
            head = entity2idx[data_sample[0].strip()]
            question = data_sample[1].strip().split(' ')
            encoded_question = [word2ix[word.strip()] for word in question]
            one_data = [torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question,dtype=torch.long), torch.tensor(len(encoded_question), dtype=torch.long), data_sample[0],data_sample[1]]
            list_data.append(one_data)
        return list_data
@app.route("/api/predict_file", methods=['POST'])
def predict_file():
    """
    预测文件测试
    :return:
    :rtype:
    """
    jsonres = request.get_json()
    test_data_path = jsonres.get('data_apth', None)
    results = model.predict_from_file()
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/predict", methods=['POST'])
def predict():
    """
    预测测试
    :return:
    :rtype:
    """
    jsonres = request.get_json()
    # data的格式
    data = jsonres.get('data', None)
    if data is None:
        return f"数据的参数data不能为空，是列表格式"
    results = model.predict(data)
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    model = LSTMKQGA()
    app.run(host='0.0.0.0', port=9966, debug=False, threaded=False)










