#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/18 10:49 上午
# @File  : predict.py.py
# @Author:
# @Desc  :  预测接口

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from model import RelationExtractor



class LSTMKQGA():
    def __init__(self, hops="1", model="ComplEx", embedding_folder="pretrained_models/embeddings/ComplEx_MetaQA_full/"):
        """

        :param hops:  1跳的问题的模型
        :type hops:
        :param model: 使用的哪个模型
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        e, r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
        # 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
        entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
        self.load_model(w_matrix,)
    def load_model(self, w_matrix, train_data_path='../../data/QA_data/MetaQA/qa_train_1hop_half.txt'):
        # 处理数据，data， list，  是【头实体，问题，答案列表】的格式，如果split是FALSE., 这里是208970条训练数据
        data = process_text_file(train_data_path, split=False)
        # 对问题进行处理，获取单词到id， id到单词的映射字典，最大长度
        word2ix, idx2word, max_len = get_vocab(data)

        # 初始化数据集
        # 初始化dataloader, batch_size: 1024
        # 初始化模型， embedding_dim： 256， hidden_dim： 200，vocab_size：117， 一共117个单词，问题的单词数量
        model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2ix),
                                  num_entities=len(idx2entity), relation_dim=relation_dim,
                                  pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop=entdrop,
                                  reldrop=reldrop, scoredrop=scoredrop, l3_reg=l3_reg, model=model_name, ls=ls,
                                  w_matrix=w_matrix, bn_list=bn_list)
        model_path = "../../checkpoints/MetaQA/best_score_model.pt"
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    def predict_from_file(self, test_data_path='../../data/QA_data/MetaQA/qa_test_1hop.txt'):
        """
        使用文件进行预测
        :param test_data_path:
        :type test_data_path:
        :return:
        :rtype:
        """

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


parser = argparse.ArgumentParser()
parser.add_argument('--ls', type=float, default=0.0, help='label_smoothing的值')
parser.add_argument('--entdrop', type=float, default=0.0, help='实体的dropout值')
parser.add_argument('--reldrop', type=float, default=0.0, help='关系的dropout值')
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0, help='学习率decay')
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0, help='修改成0，或其它数字')
parser.add_argument('--lr', type=float, default=0.0001, help="学习率")
parser.add_argument('--nb_epochs', type=int, default=90, help='训练的批次')
parser.add_argument('--gpu', type=int, default=0, help='如果使用GPU，那么使用第几个GPU')
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256, help='嵌入的维度, 问题词的嵌入的维度')
parser.add_argument('--relation_dim', type=int, default=200, help='注意，这里要和训练embedding时保持一致')
parser.add_argument('--use_cuda', type=bool, default=False, help='是否使用GPU')
parser.add_argument('--patience', type=int, default=5, help='验证集多少次不更新最好指标后，那么就停止训练')
parser.add_argument('--freeze', type=str2bool, default=True, help="是否冻结预训练的实体Embedding的参数")

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()


def prepare_embeddings(embedding_dict):
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


def get_vocab(data):
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


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
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


def validate(data_path, device, model, word2idx, entity2idx, model_name):
    model.eval()
    data = process_text_file(data_path)
    answers = []
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    for i in tqdm(range(len(data)), desc="模型评估中"):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            tail_test = torch.tensor(ans, dtype=torch.long).to(device)
            top_2 = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]
            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1
            q_text = d[-1]
            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        except:
            error_count += 1

    print(error_count)
    accuracy = total_correct / len(data)
    return answers, accuracy


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('验证集结果已经保存到文件:', fname)
    return


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()


def eval(data_path, entity_path, relation_path, entity_dict, relation_dict, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda, freeze, num_hops, entdrop, reldrop, scoredrop, l3_reg, model_name, ls, w_matrix, bn_list):
    """

    :param data_path:  '../../data/QA_data/MetaQA/qa_train_1hop.txt'
    :type data_path:  训练数据路径
    :param entity_path:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/E.npy'
    :type entity_path:  训练好的实体嵌入路径
    :param relation_path:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/R.npy'
    :type relation_path: 训练好的关系嵌入路径
    :param entity_dict:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/entities.dict'
    :type entity_dict:  实体id和实体的映射
    :param relation_dict: '../../pretrained_models/embeddings/ComplEx_MetaQA_full/relations.dict'
    :type relation_dict: 关系id和关系的映射
    :param neg_batch_size:  128
    :type neg_batch_size: int， 负样本的batch_size
    :param batch_size:  1024
    :type batch_size: int, 训练样本的batch_size
    :param shuffle: True, 是否对数据进行打乱
    :type shuffle: bool
    :param num_workers: 数据读取的并发数量, int
    :type num_workers:
    :param nb_epochs: 训练批次
    :type nb_epochs: int
    :param embedding_dim: 256， 嵌入的维度, 问题词的嵌入的维度
    :type embedding_dim: int
    :param hidden_dim: 200
    :type hidden_dim: int
    :param relation_dim: 200
    :type relation_dim: int
    :param gpu: 0, 如果使用GPU，使用第几个GPU
    :type gpu:
    :param use_cuda: False表示不使用gpu
    :type use_cuda:
    :param patience: 5， 训练指标多少个epoch不更新，就退出训练
    :type patience:
    :param freeze: True，是否冻结embedding参数
    :type freeze:
    :param validate_every: 5， 多少个epoch验证一次
    :type validate_every: int
    :param num_hops: str: '1', 跳数
    :type num_hops:
    :param lr: 学习率, 0.0001
    :type lr:
    :param entdrop: 0.0
    :type entdrop:
    :param reldrop: 0.0
    :type reldrop:
    :param scoredrop: 0.0
    :type scoredrop:
    :param l3_reg: 0.0
    :type l3_reg:
    :param model_name:  'ComplEx'
    :type model_name:
    :param decay: 1.0, 优化器的decay
    :type decay:
    :param ls: 0.0， label_smoothing
    :type ls:
    :param w_matrix:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/W.npy'
    :type w_matrix:
    :param bn_list:  batch_normalization的向量 [{'weight': array([0.9646007, 0.9430085], dtype=float32), 'bias': array([ 0.1147557 , -0.09948011], dtype=float32), 'running_mean': array([ 0.046767  , -0.05102218], dtype=float32), 'running_var': array([0.00692407, 0.00702443], dtype=float32)}, {'weight': array([1., 1.], dtype=float32), 'bias': array([0., 0.], dtype=float32), 'running_mean': array([0., 0.], dtype=float32), 'running_var': array([1., 1.], dtype=float32)}, {'weight': array([0.71846   , 0.68144286], dtype=float32), 'bias': array([-0.5668318,  0.5821315], dtype=float32), 'running_mean': array([ 0.00204753, -0.00344366], dtype=float32), 'running_var': array([0.02376127, 0.023404  ], dtype=float32)}]
    :type bn_list:
    :param valid_data_path:  '../../data/QA_data/MetaQA/qa_dev_1hop.txt'
    :type valid_data_path:
    :return:
    :rtype:
    """
    entities = np.load(entity_path)  # 实体嵌入【实体个数，嵌入维度】 ,(43234, 400)
    relations = np.load(relation_path)  # 关系嵌入 (18, 400)， 【关系种类，嵌入维度】
    # 返回e是实体对应的嵌入，字典格式， r是关系对应的嵌入，字典格式,
    e, r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
    # 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    # 处理数据，data， list，  是【头实体，问题，答案列表】的格式，如果split是FALSE., 这里是208970条训练数据
    data = process_text_file(data_path, split=False)
    # data = pickle.load(open(data_path, 'rb'))
    # 对问题进行处理，获取单词到id， id到单词的映射字典，最大长度
    word2ix, idx2word, max_len = get_vocab(data)
    # eg: cpu
    device = torch.device(gpu if use_cuda else "cpu")
    # 初始化数据集
    # 初始化dataloader, batch_size: 1024
    # 初始化模型， embedding_dim： 256， hidden_dim： 200，vocab_size：117， 一共117个单词，问题的单词数量
    model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2ix),
                              num_entities=len(idx2entity), relation_dim=relation_dim,
                              pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop=entdrop,
                              reldrop=reldrop, scoredrop=scoredrop, l3_reg=l3_reg, model=model_name, ls=ls,
                              w_matrix=w_matrix, bn_list=bn_list)
    model_path = "../../checkpoints/MetaQA/best_score_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    answers, score = validate(model=model, data_path=test_data_path, word2idx=word2ix,
                              entity2idx=entity2idx, device=device, model_name=model_name)
    print(answers)


def process_text_file(text_file, split=False):
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


def data_generator(data, word2ix, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip().split(' ')
        encoded_question = [word2ix[word.strip()] for word in question]
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question, dtype=torch.long), ans, torch.tensor(
            len(encoded_question), dtype=torch.long), data_sample[1]


eval(data_path=data_path,
      entity_path=entity_embedding_path,
      relation_path=relation_embedding_path,
      entity_dict=entity_dict,
      relation_dict=relation_dict,
      embedding_dim=args.embedding_dim,
      hidden_dim=args.hidden_dim,
      relation_dim=args.relation_dim,
      gpu=args.gpu,
      use_cuda=args.use_cuda,
      freeze=args.freeze,
      num_hops=args.hops,
      entdrop=args.entdrop,
      reldrop=args.reldrop,
      scoredrop=args.scoredrop,
      l3_reg=args.l3_reg,
      model_name=args.model,
      ls=args.ls,
      w_matrix=w_matrix,
      bn_list=bn_list)


