import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
import json
sys.path.append("../..") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

def get_vocab(data):
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
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict,'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e,r

def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('|')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G

def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key
    

def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes


def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x

def inTopk(scores, ans, k):
    """
    判断答案是否在前n个预测结果中
    :param scores:
    :type scores:
    :param ans:
    :type ans:
    :param k:
    :type k:
    :return:
    :rtype:
    """
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        x = x.item()
        if isinstance(ans, list):
            if x in ans:
                result = True
                break
        else:
            if x == ans:
                result = True
                break
    return result

def validate_v2(data_path, device, model, dataloader, entity2idx,idx2entity, model_name, writeCandidatesToFile=False):
    model.eval()
    data = process_text_file(data_path)
    answers = []
    data_gen = valid_data_tensor(data=data, dataloader=dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0
    scores_list = []
    hit_at_10 = 0
    candidates_with_scores = []
    for d in tqdm(data_gen,desc="评估"):
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check

        pred_ans = torch.argmax(scores).item()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)
        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
            # print(len(candidates))
            # exit(0)
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


        if inTopk(scores, ans, 10):
            hit_at_10 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        else:
            num_incorrect += 1
        qa_infos = d[-1]
        predict_answer_text = idx2entity[pred_ans]
        qa_infos.append(predict_answer_text)
        answers.append(qa_infos)

    print(f"hit@10值是: {hit_at_10/len(data)}")
    accuracy = total_correct/len(data)
    print(f"accuracy值是: {accuracy}")

    return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(kge_model):
    """
    获取每个实体的embedding
    :param kge_model:
    :type kge_model:
    :param hops:
    :type hops:
    :return: [number_samples, embedding_size]
    :rtype:
    """
    embedding_matrix = []
    entity2embeddings = {}
    entity2idx = {}
    entity_dict = os.path.join(dataset_path, datasetname, "entity_ids.del") # id和实体的映射
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        entity2idx[ent_name] = ent_id
        entity_embedding = embedder._embeddings(torch.LongTensor([ent_id]))[0]
        entity2embeddings[ent_name] = entity_embedding
        embedding_matrix.append(entity_embedding)
    f.close()
    idx2entity = {v: k for k, v in entity2idx.items()}
    return entity2idx, idx2entity, embedding_matrix, entity2embeddings

def train(data_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, load_from, outfile, do_batch_norm, valid_data_path=None):
    print('加载实体和关系数据')
    print("创建默认的模型")
    kge_checkpoint = load_checkpoint(load_pretained_path)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()
    print(f"加载实体的embedding和实体到id映射")
    entity2idx, idx2entity, embedding_matrix, entity2embeddings = getEntityEmbeddings(kge_model)
    print(f"实体的embedding加载完成，共有{len(entity2embeddings)}个实体")
    print(f"开始处理数据集")
    data = process_text_file(data_path)
    print(f"获取到训练集: {len(data)}")
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, entity2embeddings, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('创建模型')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, do_batch_norm=do_batch_norm)
    if load_from != '':
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    for epoch in range(nb_epochs):
        model.train()
        # model.apply(set_bn_eval)
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for i_batch, a in enumerate(loader):
            model.zero_grad()
            question_tokenized = a[0].to(device)  #问题[batch_size, max_len]
            attention_mask = a[1].to(device)    # [batch_size, max_len]
            positive_head = a[2].to(device)     # [batch_size]]
            positive_tail = a[3].to(device)    # [batch_size]
            loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
        scheduler.step()
        if epoch % validate_every == 0 and epoch != 0:
            model.eval()
            eps = 0.0001
            answers, score = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx,idx2entity=idx2entity, dataloader=dataset, device=device, model_name=model_name)
            eval_result_file = os.path.join(checkpoint_path, f"eval_result_{epoch}.json")
            with open(eval_result_file, 'w') as f:
                json.dump(answers, f, ensure_ascii=False,indent=2)
            checkpoint_path_name = os.path.join(checkpoint_path, f"{datasetname}_{epoch}_{score:.4f}.pt")
            print(f"保持checkpoint到{checkpoint_path_name}， 保持评估结果到{eval_result_file}")
            if score > best_score + eps:
                best_score = score
                no_update = 0
                best_model = model.state_dict()
                torch.save(best_model, checkpoint_path_name)
            elif (score < best_score + eps) and (no_update < patience):
                no_update +=1
                print("准确率下降了从 %f 到 %f"%(best_score, score))
            elif no_update == patience:
                print(f"模型超过了patience，停止训练，准确率为{best_score}")
                torch.save(best_model, checkpoint_path_name)
                exit()
            if epoch == nb_epochs-1:
                print("模型训练完成，准确率为 %f"%(best_score))
                torch.save(best_model, checkpoint_path_name)

def process_text_file(text_file):
    """
    处理数据集
    :param text_file:
    :type text_file:
    :param split:
    :type split:
    :return: 头实体，问题，答案
    :rtype:
    """
    data_array = []
    with open(text_file, 'r') as f:
        data = json.load(f)
    for d in data:
        data_array.append(list(d.values()))
    return data_array

def valid_data_tensor(data, dataloader, entity2idx):
    tensor_data = []
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]  # 答案转换成id
        else:
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]
        head = torch.tensor(head, dtype=torch.long)
        tensor_data.append([head, question_tokenized, attention_mask, ans, data_sample])
    return tensor_data

if __name__ == '__main__':
    project_path = "../../"
    parser = argparse.ArgumentParser()
    parser.add_argument('--hops', type=str, default='1')
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--ls', type=float, default=0.3, help='label smoothing')
    parser.add_argument('--validate_every', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--model', type=str, default='MYMODEL', help='Bert后面接的模型')
    parser.add_argument('--mode', type=str, default='train', help='train/eval， 二选一')
    parser.add_argument('--outfile', type=str, default='best_score_model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--entdrop', type=float, default=0.0)
    parser.add_argument('--reldrop', type=float, default=0.0)
    parser.add_argument('--scoredrop', type=float, default=0.0)
    parser.add_argument('--l3_reg', type=float, default=0.0)
    parser.add_argument('--decay', type=float, default=0.3)
    parser.add_argument('--shuffle_data', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--nb_epochs', type=int, default=90)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--neg_batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--relation_dim', type=int, default=100,help="保持和实体的embedding一致")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--freeze', type=str2bool, default=True)
    parser.add_argument('--do_batch_norm', type=str2bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(project_path, "checkpoints"))
    parser.add_argument('--pretrain_model_path', type=str, default=os.path.join(project_path, "pretrained_models"))
    parser.add_argument('--datasetname', type=str, default="mydata", help="选择使用哪个数据集，默认fbwq,也可以是MetaQA, mydata")
    parser.add_argument('--dataset_path', type=str, default=os.path.join(project_path, "data"), help="数据集路径")
    args = parser.parse_args()
    datasetname = args.datasetname
    dataset_path = args.dataset_path
    load_pretained_path = f"{args.pretrain_model_path}/{datasetname}/kge.pt"
    checkpoint_path = args.checkpoint_path

    hops = args.hops
    model_name = args.model

    if datasetname == "MetaQA":
        data_path = '../../data/QA_data/MetaQA/qa_train_1hop.txt'
        valid_data_path = '../../data/QA_data/MetaQA/qa_dev_1hop.txt'
        test_data_path = '../../data/QA_data/MetaQA/qa_test_1hop.txt'
    elif datasetname =="fbwq":
        data_path = '../../data/QA_data/MetaQA/qa_train_1hop.txt'
        valid_data_path = '../../data/QA_data/MetaQA/qa_dev_1hop.txt'
        test_data_path = '../../data/QA_data/MetaQA/qa_test_1hop.txt'
    elif datasetname == "mydata":
        data_path = '../../data/mydata/train.json'
        valid_data_path = '../../data/mydata/valid.json'
        test_data_path = '../../data/mydata/test.json'
    else:
        raise Exception("不支持的数据集")

    train(data_path=data_path,
    neg_batch_size=args.neg_batch_size,
    batch_size=args.batch_size,
    shuffle=args.shuffle_data,
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    relation_dim=args.relation_dim,
    gpu=args.gpu,
    use_cuda=args.use_cuda,
    valid_data_path=valid_data_path,
    patience=args.patience,
    validate_every=args.validate_every,
    freeze=args.freeze,
    hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    load_from=args.load_from,
    outfile=args.outfile,
    do_batch_norm=args.do_batch_norm)
