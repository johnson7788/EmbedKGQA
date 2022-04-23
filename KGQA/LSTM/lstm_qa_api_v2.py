#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/13 3:10 下午
# @File  : lstm_qa_api.py
# @Author: jinxia
# @Contact : github: jinxia
# @Desc  :  LSTM+ComplEx 问答模型的服务端

import json
import os
import re
import pandas as pd
import requests
import torch
import logging.config
from tqdm import tqdm
import numpy as np
from model import RelationExtractor
from flask import Flask, request, jsonify, abort
import random
from py2neo import Graph

import logging.config
# from py2neo import Graph
# from flask_cors import CORS
# from flask import Flask, request, jsonify, abort
# app = Flask(__name__)
# CORS(app, supports_credentials=True)

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

class LSTMKQGA(object):
    def __init__(self, model_name="ComplEx", embedding_folder="../../kg_embeddings/waishen/"):
        """
        :param model_name: 使用的哪个模型
        """
        self.embedding_dim = 256
        self.hidden_dim = 200
        self.relation_dim = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "../../checkpoints/waishen/best_score_model.pt"
        train_data_path = '../../data/waishen/qa_train.json'
        entity_embedding_path = embedding_folder + '/E.npy'
        relation_embedding_path = embedding_folder + '/R.npy'
        entity_dict = embedding_folder + '/entities.dict'
        relation_dict = embedding_folder + '/relations.dict'
        w_matrix = embedding_folder + '/W.npy'
        bn_list = []

        for i in range(3):
            bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
            bn_list.append(bn.item())
        # 实体嵌入【实体个数，嵌入维度】 ,(43234, 400)
        entities = np.load(entity_embedding_path)
        # 关系嵌入 (18, 400)， 【关系种类，嵌入维度】
        relations = np.load(relation_embedding_path)
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
            for word in sent:
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

    def process_text_file(self,text_file, split=False):
        '''训练问题的数据进行处理，['[市审计局]审计了哪个单位', '上海理工大学'] 变成-》['市审计局', '空审计了哪个单位', ['上海理工大学']]，将问题变成关系'''
        data_array = []
        if text_file.endswith('.json'):
            with open(text_file, 'r') as f:  # 打开问答训练数据 qa_train_1hop.json
                data = json.load(f)  # data 900个问答对
                for one_data in data:  # one_data = ['[广州市审计局]审计了哪个单位', '广州市现代农业发展平台建设（示范园区建设）资金']
                    # question: ['', '广州市审计局]审计了哪个单位']
                    question = one_data[0].split('[')
                    # question_1:''
                    question_1 = question[0]
                    # question_2: ['广州市审计局', '审计了哪个单位']
                    question_2 = question[1].split(']')
                    # head: '广州市审计局' 字符串格式
                    head = question_2[0].strip()
                    # question_2: '审计了哪个单位'
                    question_2 = question_2[1]
                    # 问题是英文时候 question: 'what movies are about NE'
                    # question = question_1 + 'NE' + question_2
                    # 问题是中文时候 question: '空审计了哪个单位'
                    question = question_1 + '空' + question_2
                    ans = one_data[1].split('|')  # ans:['广州市现代农业发展平台建设（示范园区建设）资金']
                    data_array.append([head, question.strip(), ans])
            return data_array

        else:
            data_file = open(text_file, 'r')
            # data_array = []
            for data_line in data_file.readlines():  # data_line:  what movies are about [ginger rogers]	Top Hat|Kitty Foyle|The Barkleys of Broadway
                data_line = data_line.strip()
                if data_line == '':
                    continue
                data_line = data_line.strip().split(
                    '\t')  # data_line:['what movies are about [ginger rogers]', 'Top Hat|Kitty Foyle|The Barkleys of Broadway']
                question = data_line[0].split('[')  # question: ['what movies are about ', 'ginger rogers]']
                question_1 = question[0]  # question_1:'what movies are about '
                question_2 = question[1].split(']')  # question_2: ['ginger rogers', '']
                head = question_2[0].strip()  # head: 'ginger rogers'
                question_2 = question_2[1]  # question_2: ''
                question = question_1 + 'NE' + question_2  # question: 'what movies are about NE'
                ans = data_line[1].split('|')
                data_array.append([head, question.strip(),
                                   ans])  # data_arrray:['ginger rogers', 'what movies are about NE', ['Top Hat', 'Kitty Foyle', 'The Barkleys of Broadway']]
            if split == False:
                return data_array
            else:
                data = []
                for line in data_array:
                    head = line[0]
                    question = line[1]
                    tails = line[2]
                    for tail in tails:
                        data.append([head, question, tail])
                return data

    def data_generator(self,data, word2ix, entity2idx):
        list_data = []
        for i in range(len(data)):
            # data_sample['市粮食局', '空的简称是什么', ['现代商务']]
            data_sample = data[i]
            # head，头实体id：675
            head = entity2idx[data_sample[0].strip()]
            # question:['空', '的', '简', '称', '是', '什', '么']
            question = list(data_sample[1].strip())
            # 问题编码encoded_question：[0, 8, 19, 20, 10, 11, 12]
            encoded_question = [word2ix[word.strip()] for word in question]
            one_data =  [torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question,dtype=torch.long), torch.tensor(len(encoded_question), dtype=torch.long), data_sample[0],data_sample[1]]
            list_data.append(one_data)
        return list_data

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
            quesiton = question_text.replace('空', f"{qestion_head}")
            # answers.append(f"问题是:{quesiton}, 答案是: {answer_text}")
            answers.append(answer_text)
        return answers

    def validate(self,data_path, device, model, word2idx, entity2idx, model_name):

        model.eval()
        # data_path：开发集的路径qa_dev_1hop.json 返回data有114条
        data = self.process_text_file(data_path)
        answers = []
        # data_generator 生成器，调用的时候才进入
        data_gen = self.data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
        total_correct = 0
        error_count = 0
        for i in tqdm(range(len(data))):
            try:
                # 调用生成器
                d = next(data_gen)
                # 头实体id head:tensor(675, device='cuda:0')
                head = d[0].to(device)
                # 问题编码：question:tensor([ 0,  8, 19, 20, 10, 11, 12], device='cuda:0')
                question = d[1].to(device)
                # 真实答案id ans:881
                ans = d[2]
                # 问题长度 ques_len：7
                ques_len = d[3].unsqueeze(0)
                # 问题的id转成tensor：tail_test：tensor([881], device='cuda:0')
                tail_test = torch.tensor(ans, dtype=torch.long).to(device)
                # 通过头实体，问题，问题长度，计算尾实体id tensor(881，680)
                top_2 = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
                # tensor 转成list格式：top_2_idx：[881, 680]
                top_2_idx = top_2[1].tolist()[0]
                # 头实体id转成int格式 head_idx：675
                head_idx = head.tolist()
                # 如果头实体id = top_2_idx[0]，那么预测尾实体id=top_2_idx[1]。否则预测尾实体id=top_2_idx[0]=881，正确的
                if top_2_idx[0] == head_idx:
                    pred_ans = top_2_idx[1]
                else:
                    pred_ans = top_2_idx[0]
                if type(ans) is int:
                    ans = [ans]
                is_correct = 0
                # 如果预测尾实体id在真实尾实体id中，total_correct，正确值+1
                if pred_ans in ans:
                    total_correct += 1
                    is_correct = 1
                else:
                    error_count += 1
                # 问题 q_text：
                q_text = d[-1]
                # 答案组成：问题，预测尾实体答案id，答案正确为1，错误为0
                answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
            except Exception as e:
                print(f"错误的异常是:")
                print(e)
                error_count += 1

        print('回答问题错误的的个数是：', error_count)
        # 准确率计算：总正确/总数据
        accuracy = total_correct / len(data)
        return answers, accuracy

def verify_data_format(data=None, text=None):
    """
    判断data是否是 实体+ 问题的标准格式，如果不是，那么识别实体，返回实体+问题的标准格式
    """
    #对问题进行实体识别，进入是实体识别预测模型
    nerurl = "http://127.0.0.1:3326/api/label_studio_shenjiner_predict"
    headers = {'content-type': 'application/json'}
    if data:
        if isinstance(data[0], list) and len(data[0]) == 2:
            print("是标准格式，不需要实体识别")
            return data
        else:
            new_data = []
            print("需要实体识别，调用/api/shenjiner_predict")
            params = {'data': data}
            r = requests.post(nerurl, headers=headers, data=json.dumps(params), timeout=360)
            result = r.json()
            # 取预测的第一个实体结果作为头实体
            for one_data, res in zip(data, result):
                question = one_data
                entity_info = res[0]
                entity = entity_info[0]
                start = entity_info[3]
                end = entity_info[4]
                qes_text = question[0:start] + "空" + question[end:]
                one = [entity, qes_text]
                new_data.append(one)
        return new_data
    else:
        new_data = []
        print("需要实体识别，调用/api/shenjiner_predict")
        # params = {'data': text}
        params = {'data': [text]}
        r = requests.post(nerurl, headers=headers, data=json.dumps(params), timeout=360)
        result = r.json()
        res = result[0]
        question = text
        entity_info = res[0]
        entity = entity_info[0]
        start = entity_info[3]
        end = entity_info[4]
        qes_text = question[0:start] + "空" + question[end:]
        one = [entity, qes_text]
        new_data.append(one)
        return new_data

def get_data():
    """
    生成三元组数据集，并保存到本地
    :param use_cache: 判断是否使用本地的json文件还是重新获取
    :return:
    """
    # 使用数据库，连接的用户是neo4j，连接密码welcome，连接的数据库是city
    graph = Graph(host='192.168.10.242', user='neo4j', password='welcome', name='neo4j', port=7687)  #
    ralations = ['审计', '子单位', '涉及', '简称', '存在', '审计日期','篇章','条款']
    sanyuan = []
    for i in range(len(ralations)):
        one_relations = ralations[i]
        # 运行CQL语句，把所有的关系都查询出来
        sql = """ MATCH p = (()-[r:%s]->()) RETURN p """ % (one_relations)
        # sql = """CREATE CONSTRAINT ON (c:%s) ASSERT c.%s IS UNIQUE""" % (label_name, unique)

        res = graph.run(sql)
        res_list = list(res)
        for info in res_list:
            record = info[0]
            head = record.nodes[0]
            tail = record.nodes[1]
            relation = record.relationships
            relaion0 = relation[0]
            rel_name = type(relaion0).__name__
            # print(f"头实体信息: {head['name']}")
            # print(f"尾实体信息: {tail['name']}")
            # print(f"关系是: {rel_name}")
            one_sanyuan = [head['name'], rel_name, tail['name']]
            sanyuan.append(one_sanyuan)
    return sanyuan

def generate_question_answer(data_dir):
    """
    根据三元组，生成问答数据集
    :return:
    """
    # 将json文件的内容导出列表
    # path = '/Users/admin/Desktop/words2.json'
    data  = get_data()
    ralations = ['审计', '子单位', '涉及', '简称', '存在', '审计日期', '篇章', '条款']
    qa = []
    # with open(path, 'r') as f:
    #     data = json.load(f)

    for one_data in data:
        if one_data[1] == '审计':
            shen = '[' + one_data[0] + ']' + "审计了哪个单位"
            qa_shen = [shen, one_data[2]]
            print(qa_shen)
            qa.append(qa_shen)

        elif one_data[1] == "子单位":
            zi = '[' + one_data[0] + ']' + "的子单位是什么"
            qa_zi = [zi, one_data[2]]
            print(qa_zi)
            qa.append(qa_zi)

        elif one_data[1] == "涉及":
            she = '[' + one_data[0] + ']' + "涉及的资金是多少"
            qa_she = [she, one_data[2]]
            print(qa_she)
            qa.append(qa_she)

        elif one_data[1] == "简称":
            jian = '[' + one_data[0] + ']' + "的简称是什么"
            qa_jian = [jian, one_data[2]]
            print(qa_jian)
            qa.append(qa_jian)

        elif one_data[1] == "存在":
            cun = '[' + one_data[0] + ']' + "存在哪些审计问题"
            qa_cun = [cun, one_data[2]]
            print(qa_cun)
            qa.append(qa_cun)

        elif one_data[1] == "篇章":
            cun = '[' + one_data[0] + ']' + "有哪些篇章"
            qa_cun = [cun, one_data[2]]
            print(qa_cun)
            qa.append(qa_cun)

        elif one_data[1] == "条款":
            cun = '[' + one_data[0] + ']' + "有哪些条款"
            qa_cun = [cun, one_data[2]]
            print(qa_cun)
            qa.append(qa_cun)


    print(f"生成的问答对数据集大小：{len(qa)}")
    train_data_num = int(len(qa) * 0.8)
    valid_data_num = int(len(qa) * 0.1)
    train_data, valid_data, test_data = qa[:train_data_num], qa[train_data_num:train_data_num + valid_data_num], qa[
                                                                                                                 train_data_num + valid_data_num:]
    # 保存到json格式的文件中
    with open(os.path.join(data_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(os.path.join(data_dir, 'valid.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False)
    with open(os.path.join(data_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    print("生成问答数据集完成")


class PreprocessTrain():
    def __init__(self):
        graph = Graph(host='192.168.10.242', user='neo4j', password='welcome', name='neo4j', port=7687)  #
        self.graph = graph
        self.project_dir = "/home/fujinxia/fu/EmbedKGQA-master"
        self.data_dir = os.path.join(self.project_dir, "data/waishen")
        self.all_data_json = os.path.join(self.data_dir, "all.json")
        self.relations = ['审计', '子单位', '涉及', '简称', '存在', '审计日期', '篇章', '条款']

    def data_triple(self):
        # graph = Graph(host='127.0.0.1', user='neo4j', password='welcome', name='neo4j',port=7698)#
        # 所有的关系：
        # ralations = ['一级分类','业务来源','二级分类','审计项目','所属专业','数据来源']
        # ralations = ['审计', '子单位', '涉及', '简称', '存在', '篇章', '条款']
        # ralations = ['审计', '子单位', '涉及', '简称', '存在']
        sanyuan = []
        for i in range(len(self.relations)):
            one_relations = self.relations[i]
            # 运行CQL语句，把所有的关系都查询出来
            sql = """ MATCH p = (()-[r:%s]->()) RETURN p """ % (one_relations)
            # sql = """CREATE CONSTRAINT ON (c:%s) ASSERT c.%s IS UNIQUE""" % (label_name, unique)

            res = self.graph.run(sql)
            res_list = list(res)
            for info in res_list:
                record = info[0]
                head = record.nodes[0]
                tail = record.nodes[1]
                relation = record.relationships
                relaion0 = relation[0]
                rel_name = type(relaion0).__name__
                # print(f"头实体信息: {head['name']}")
                # print(f"尾实体信息: {tail['name']}")
                # print(f"关系是: {rel_name}")
                one_sanyuan = [head['name'], rel_name, tail['name']]
                sanyuan.append(one_sanyuan)

        print(f"生成的问答对数据集大小：{len(sanyuan)}")
        random.shuffle(sanyuan)
        train_data_num = int(len(sanyuan) * 0.8)
        valid_data_num = int(len(sanyuan) * 0.1)
        train_data, valid_data, test_data = sanyuan[:train_data_num], sanyuan[train_data_num:train_data_num + valid_data_num], sanyuan[train_data_num + valid_data_num:]
        # 保存到json格式的文件中
        with open(os.path.join(self.data_dir, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(os.path.join(self.data_dir, 'valid.json'), 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, ensure_ascii=False)
        with open(os.path.join(self.data_dir, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
        print("生成问答数据集完成")

        # print(len(sanyuan))
        # 将列表的内容导入json文件
        with open(self.all_data_json, 'w') as f:
            json.dump(sanyuan, f, ensure_ascii=False, indent=2)
        print(f'将全部数据已经保存到: {self.all_data_json}')

    def data_qa(self):
        qa = []
        with open(self.all_data_json, 'r') as f:
            data = json.load(f)
            for one_data in data:
                if one_data[1] == '审计':
                    shen = '[' + one_data[0] + ']' + "审计了哪个单位"
                    qa_shen = [shen, one_data[2]]
                    qa.append(qa_shen)

                elif one_data[1] == "子单位":
                    zi = '[' + one_data[0] + ']' + "的子单位是什么"
                    qa_zi = [zi, one_data[2]]
                    qa.append(qa_zi)

                elif one_data[1] == "涉及":
                    she = '[' + one_data[0] + ']' + "涉及的资金是多少"
                    qa_she = [she, one_data[2]]
                    qa.append(qa_she)

                elif one_data[1] == "简称":
                    jian = '[' + one_data[0] + ']' + "的简称是什么"
                    qa_jian = [jian, one_data[2]]
                    qa.append(qa_jian)

                elif one_data[1] == "存在":
                    cun = '[' + one_data[0] + ']' + "存在哪些审计问题"
                    qa_cun = [cun, one_data[2]]
                    qa.append(qa_cun)

                elif one_data[1] == "篇章":
                    cun = '[' + one_data[0] + ']' + "有哪些篇章"
                    qa_cun = [cun, one_data[2]]
                    qa.append(qa_cun)

                elif one_data[1] == "条款":
                    cun = '[' + one_data[0] + ']' + "有哪些条款"
                    qa_cun = [cun, one_data[2]]
                    qa.append(qa_cun)
            print(f"生成的问答对数据集大小：{len(qa)}")
            random.shuffle(qa)
            train_data_num = int(len(qa) * 0.8)
            valid_data_num = int(len(qa) * 0.1)
            train_data, valid_data, test_data = qa[:train_data_num], qa[
                                                                     train_data_num:train_data_num + valid_data_num], qa[
                                                                                                                      train_data_num + valid_data_num:]
            # 保存到json格式的文件中
            with open(os.path.join(self.data_dir, 'qa_train.json'), 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False)
            with open(os.path.join(self.data_dir, 'qa_valid.json'), 'w', encoding='utf-8') as f:
                json.dump(valid_data, f, ensure_ascii=False)
            with open(os.path.join(self.data_dir, 'qa_test.json'), 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False)
            print("生成问答数据集完成")

    def train_embedding(self):
        cmd = f"cd {self.project_dir}/train_embeddings && python main.py"
        os.system(command=cmd)

    def train_kgqa(self):
        cmd = f"cd {self.project_dir}/KGQA/LSTM && python main.py"
        os.system(command=cmd)

    def run(self):
        logging.info(f"正在从neo4j获取数据，生成训练数据")
        self.data_triple()
        logging.info(f"正在使用生成的数据生成问答对数")
        self.data_qa()
        logging.info(f"正在进行训练知识图谱嵌入")
        self.train_embedding()
        logging.info(f"正在进行知识问答模型训练")
        self.train_kgqa()

@app.route("/api/qa_predict", methods=['POST'])
def lstm_predict():
    '''
    各一个问题，预测答案
    eg:
        问题:"[市财政局]存在哪些审计问题"
        答案："国有资本经营预算审计"
    '''

    jsonres = request.get_json()
    # data = {
    #     "sender": "0001_2",
    #     "message": "wenti"
    # }
    # # data是列表格式
    # data = jsonres.get('data', None)
    # # text需要是一句话，字符串格式
    # text = jsonres.get('text', None)
    # if data is None and text is None:
    #     return f"数据的参数data和text不能同时为空，data是列表格式，text是字符串格式"
    # if data:
    #     data = verify_data_format(data=data)
    # else:
    #     data = verify_data_format(text=text)


    #从neo4j中抽取三元组
    #三元组生成问题
    # generate_question_answer(data_dir='')
    #将问题进行训练

    text = jsonres['message']
    id = jsonres['sender']
    # data = jsonres.get('data1')
    #text是字符串格式，进入实体识别预测
    try:
        data = verify_data_format(text=text)
        print(f"数据是: {data}")
        model_results = model.predict(data)
        # if data:
        #     # 只有一条数据
        #当不能回答问题时，直接返回：该问题无法回答
        result1 = model_results[0]
    except Exception as e:
        result1 = "该问题暂时无法回答"
        print(f"问题: {text}无法回答，报错如下: ")
        print(e)
    results = [
                {
                    "recipient_id": id,
                    "custom": {
                        "type": "text",
                        "content":result1
                    }
                }
            ]


    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/train", methods=['POST'])
def do_train():
    '''
    训练模型
    curl -X POST http://127.0.0.1:9966/api/train
    eg:
        问题:"[市财政局]存在哪些审计问题"
        答案："国有资本经营预算审计"
    '''
    logger.info(f"调用了训练")
    global model
    model = LSTMKQGA()
    return jsonify("训练完成")

if __name__ == "__main__":
    model = LSTMKQGA()
    PT = PreprocessTrain()
    PT.run()
    app.run(host='0.0.0.0', port=9966, debug=False, threaded=False)