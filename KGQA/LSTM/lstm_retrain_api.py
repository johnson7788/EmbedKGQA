#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/23 2:04 下午
# @File  : lstm_retrain_api.py
# @Author: jinxia
# @Contact : github: jinxia
# @Desc  :  当图谱有实体更新时，需要重新训练问答模型，当实体类型更新需要重新训练预测模型和问答模型

#1 从neo4j中取出三元组
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

class PreprocessTrain():
    def __init__(self):
        graph = Graph(host='192.168.10.242', user='neo4j', password='welcome', name='neo4j', port=7687)  #
        self.graph = graph
        self.project_dir = "/EmbedKGQA-master"
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
        # train_data_num = int(len(sanyuan) * 0.8)
        train_data_num = int(len(sanyuan))
        valid_data_num = int(len(sanyuan) * 0.1)
        train_data, valid_data, test_data = sanyuan[:train_data_num], sanyuan[train_data_num:train_data_num + valid_data_num], sanyuan[
                                                                                                                     train_data_num + valid_data_num:]
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
        self.data_triple()
        self.data_qa()
        self.train_embedding()
        self.train_kgqa()

if __name__ == '__main__':
    PT = PreprocessTrain()
    app.run(host='0.0.0.0', port=9966, debug=False, threaded=False)



