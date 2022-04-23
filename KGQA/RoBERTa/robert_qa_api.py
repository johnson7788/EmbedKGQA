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
from pruning_model import *
from pruning_dataloader import DatasetPruning, DataLoaderPruning
from pruning_model import PruningModel
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
from model import RelationExtractor
from flask import Flask, request, jsonify, abort

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

class RoBERTaKQGA(object):

    def __init__(self, model_name="ComplEx"):
        """
        :param model_name: 使用的哪个模型
        """
        self.ls = 0.1
        self.hidden_dim = 256
        self.relation_dim = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def process_data_file(fname, rel2idx, idx2rel):

        f = open(fname, 'r')
        data = []
        # line字符串格式：'what is the name of justin bieber brother [m.06w2sn5]	people.sibling_relationship.sibling|people.person.sibling_s
        for line in f:
            # line转成列表格式：['what is the name of justin bieber brother [m.06w2sn5]', 'people.sibling_relationship.sibling|people.person.sibling_s']
            line = line.strip().split('\t')
            # question：'what is the name of justin bieber brother [m.06w2sn5]'
            question = line[0].strip()
            # TODO only work for webqsp. to remove entity from metaqa, use something else
            # 把[]中实体从问题中移除 question：'what is the name of justin bieber brother '
            question = question.split('[')[0]
            # 答案列表：rel_list：['people.sibling_relationship.sibling', 'people.person.sibling_s']
            rel_list = line[1].split('|')
            # 答案列表的id格式 rel_id_list：[7242, 7212]
            rel_id_list = []
            for rel in rel_list:
                rel_id_list.append(rel2idx[rel])
            # data：[('what is the name of justin bieber brother ', [7242, 7212], 'what is the name of justin bieber brother [m.06w2sn5]')]
            data.append((question, rel_id_list, line[0].strip()))  # data：('what is the name of justin bieber brother ', [886, 880], 'what is the name of justin bieber brother [m.06w2sn5]')
        return data


    def predict(self, data):
        """
        data:
        :param data:  每条数据都是[头实体，问题，尾实体列表] 的格式, eg: ['Grégoire Colin', 'what does NE appear in', ['Before the Rain']]
        :type data:
        :return:
        :rtype:
        """

    f = open('/home/wac/johnson/kg/EmbedKGQA-master/data/fbwq_full/relations_all.dict', 'r')
    # 获取关系-id的字典格式 1144个
    rel2idx = {}
    # 获取id-关系的字典格式 1144个
    idx2rel = {}
    # line：american_football.football_coach.coaching_history	2
    for line in f:
        # line：['american_football.football_coach.coaching_history', '2']
        line = line.strip().split('\t')
        # id:2
        id = int(line[1])
        # rel字符串:'american_football.football_coach.coaching_history'
        rel = line[0]
        # 答案：id 字典 一共18478
        rel2idx[rel] = id
        # di：答案 字典
        idx2rel[id] = rel
    f.close()
    # 模型的初始化
    model = PruningModel(rel2idx, idx2rel, ls=self.ls)
    checkpoint_file = "checkpoints/pruning/best_score_model.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)
    print('loaded from ', checkpoint_file)
    model.to(device)
    # 处理问题数据集3036个问题，问题变成：[('what is the name of justin bieber brother ', [7242, 7212], 'what is the name of justin bieber brother [m.06w2sn5]')]
    new_data = process_data_file(data, rel2idx, idx2rel)
    for i in tqdm(range(len(new_data))):
        # try:
        d = new_data[i]

        question = d[0]

        question_tokenized, attention_mask = new_data.tokenize_question(question)

        question_tokenized = question_tokenized.to(self.device)

        attention_mask = attention_mask.to(self.device)

        rel_id_list = d[1]
        # 得出预测值（18478，）
        prediction = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        # top2：tensor([6058], device='cuda:0')
        top2 = torch.topk(prediction, 1)
        # 获得三元组答案
        top2 = top2[1]




def verify_data_format(text):
    '''对输入的问题进行预处理，生成的格式为：
    what kind of money to take to bahamas [m.0160w]	location.country.currency_used'''
    pass



@app.route("/api/qa_predict_robert", methods=['POST'])
def robert_predict():
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

    text = jsonres['message']
    id = jsonres['sender']
    # data = jsonres.get('data1')
    #对输入的问题进行处理，生成data：what kind of money to take to bahamas [m.0160w]	location.country.currency_used
    data = verify_data_format(text=text)
    print(f"数据是: {data}")
    #对答案进行预测
    results = model.predict(data)
    # if data:
    #     # 只有一条数据
    result1 = results[0]
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


if __name__ == "__main__":
    model = RoBERTaKQGA()
    app.run(host='0.0.0.0', port=9966, debug=False, threaded=False)