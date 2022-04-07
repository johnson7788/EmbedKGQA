import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--validate_every', type=int, default=5)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--kg_type', type=str, default='full')

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0)#设置工作的线程，多进程debug是，会卡死
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=90)#epoch次数
parser.add_argument('--gpu', type=int, default=0)#选择第几块显卡
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--relation_dim', type=int, default=200) #关系的维度保持200
parser.add_argument('--use_cuda', type=bool, default=True) #是否使用GPU
parser.add_argument('--patience', type=int, default=5)#验证集/开发集5次不更新最好指标后，就停止训练
parser.add_argument('--freeze', type=str2bool, default=True)#冻结embedding预训练的参数

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
            sent = d[1]
            # for word in sent.split(): #问题是英文的时候
            for word in sent: #问题是中文的时候
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
        line = line.strip().split('\t') #中文的时候屏蔽
        # line = line.strip().split('  ') #中文的时候屏蔽

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


def validate(data_path, device, model, word2idx, entity2idx, model_name):
    model.eval()
    data = process_text_file(data_path)
    answers = []
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    for i in tqdm(range(len(data))):
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
    accuracy = total_correct/len(data)
    return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('验证集的答案写入 ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()


def train(data_path, entity_path, relation_path, entity_dict, relation_dict, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, num_hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, w_matrix, bn_list, valid_data_path=None):
    entities = np.load(entity_path) #entities:(1010,400)  400是啥？？entity_path = '../../kg_embeddings/waishen//E.npy'
    relations = np.load(relation_path) #relations:(12,400)relation_path = '../../kg_embeddings/waishen//R.npy'
    e,r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations) #获取实体和关系的字典格式 e dict=1003, r dict=12
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e) #转换实体的embedding entity2idx dict=1000
    data = process_text_file(data_path, split=False) #训练问题的数据进行处理，
    # data = pickle.load(open(data_path, 'rb'))
    word2ix,idx2word, max_len = get_vocab(data) #处理后的问题数据，进行编码
    hops = str(num_hops)
    # print(idx2word)
    # aditay
    # print(idx2word.keys())
    device = torch.device(gpu if use_cuda else "cpu") #device：cuda:0
    dataset = DatasetMetaQA(data=data, word2ix=word2ix, relations=r, entities=e, entity2idx=entity2idx)
    data_loader = DataLoaderMetaQA(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2ix), num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, w_matrix = w_matrix, bn_list=bn_list)
    model.to(device) #关系提取器，设置模型的训练的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #设置优化器
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    for epoch in range(nb_epochs): #开始训练 ，validate_every=5 不知道什么意思
        phases = []  #phase：['train', 'train', 'train', 'train', 'train', 'valid']训练5次，评估1次
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                if freeze == True:
                    # print('Freezing batch norm layers')
                    model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question = a[0].to(device) #questions：（128，11）（128，8）
                    sent_len = a[1].to(device)#sent_len:128
                    positive_head = a[2].to(device) #(128)
                    positive_tail = a[3].to(device)       #(128,43234)（128，1000）
                    #
                    loss = model(sentence=question, p_head=positive_head, p_tail=positive_tail, question_len=sent_len)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                
                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                answers, score = validate(model=model, data_path= valid_data_path, word2idx= word2ix, entity2idx= entity2idx, device=device, model_name=model_name)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print(hops + " hop Validation accuracy increased from previous epoch", score)
                    _, test_score = validate(model=model, data_path= test_data_path, word2idx= word2ix, entity2idx= entity2idx, device=device, model_name=model_name)
                    print('验证集最好的结果:', test_score)
                    writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    suffix = ''
                    if freeze == True:
                        suffix = '_frozen'
                    checkpoint_path = '../../checkpoints/MetaQA/'
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    checkpoint_file_name = checkpoint_path +model_name+ '_' + num_hops + suffix + ".pt"
                    print('Saving checkpoint to ', checkpoint_file_name)
                    torch.save(model.state_dict(), checkpoint_file_name)
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    torch.save(best_model, checkpoint_path+ "best_score_model.pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stopping and saving model.")
                    torch.save(best_model, checkpoint_path +"best_score_model.pt")
                    exit()
                    

def process_text_file(text_file, split=False):
    data_array = []
    if text_file.endswith('.json'):
        with open(text_file, 'r') as f:
            data = json.load(f)
            for one_data in data:
                question = one_data[0].split('[')  # question: ['what movies are about ', 'ginger rogers]']
                question_1 = question[0]  # question_1:'what movies are about '
                question_2 = question[1].split(']')  # question_2: ['ginger rogers', '']
                head = question_2[0].strip()  # head: 'ginger rogers'
                question_2 = question_2[1]  # question_2: ''
                # question = question_1 + 'NE' + question_2  # 问题是英文时候 question: 'what movies are about NE'
                question = question_1 + '空' + question_2  # 问题是中文时候 question: 'what movies are about NE'
                ans = one_data[1].split('|')
                data_array.append([head, question.strip(), ans])
        return data_array

    else:
        data_file = open(text_file, 'r')
        # data_array = []
        for data_line in data_file.readlines():#data_line:  what movies are about [ginger rogers]	Top Hat|Kitty Foyle|The Barkleys of Broadway
            data_line = data_line.strip()
            if data_line == '':
                continue
            data_line = data_line.strip().split('\t') #data_line:['what movies are about [ginger rogers]', 'Top Hat|Kitty Foyle|The Barkleys of Broadway']
            question = data_line[0].split('[') #question: ['what movies are about ', 'ginger rogers]']
            question_1 = question[0] #question_1:'what movies are about '
            question_2 = question[1].split(']') #question_2: ['ginger rogers', '']
            head = question_2[0].strip() #head: 'ginger rogers'
            question_2 = question_2[1] #question_2: ''
            question = question_1+'NE'+question_2 #question: 'what movies are about NE'
            ans = data_line[1].split('|')
            data_array.append([head, question.strip(), ans]) #data_arrray:['ginger rogers', 'what movies are about NE', ['Top Hat', 'Kitty Foyle', 'The Barkleys of Broadway']]
        if split==False:
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

        yield torch.tensor(head, dtype=torch.long),torch.tensor(encoded_question, dtype=torch.long) , ans, torch.tensor(len(encoded_question), dtype=torch.long), data_sample[1]




# hops = args.hops #1，2，3跳问题
# if hops in ['1', '2', '3']:
#     hops = hops + 'hop'
# if args.kg_type == 'half':
#     data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '_half.txt'
# else:
#     data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '.txt'
# print('Train file is ', data_path)
#
# hops_without_old = hops.replace('_old', '')
# valid_data_path = '../../data/QA_data/MetaQA/qa_dev_' + hops_without_old + '.txt'
# test_data_path = '../../data/QA_data/MetaQA/qa_test_' + hops_without_old + '.txt'

#mydata训练数据的路径
data_path = '../../data/QA_data/waishen/qa_train_1hop.json'
print('Train file is ', data_path)
valid_data_path = '../../data/QA_data/waishen/qa_dev_1hop.json'
test_data_path = '../../data/QA_data/waishen/qa_test_1hop.json'


# model_name = args.model
# kg_type = args.kg_type
# print('KG type is', kg_type)
# # embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_' + kg_type
# embedding_folder = '../../kg_embeddings/MetaQA/'

#mydata图谱嵌入生成数据的路径
embedding_folder = '../../kg_embeddings/waishen/'

entity_embedding_path = embedding_folder + '/E.npy'
relation_embedding_path = embedding_folder + '/R.npy'
entity_dict = embedding_folder + '/entities.dict'
relation_dict = embedding_folder + '/relations.dict'
w_matrix =  embedding_folder + '/W.npy'  #没有这个文件 W.npy

bn_list = [] #bn不知道什么什么文件？w 和 b 的值，模型参数，批归一化的时候用

for i in range(3):
    bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
    bn_list.append(bn.item())

if args.mode == 'train':
    train(data_path=data_path, 
    entity_path=entity_embedding_path, 
    relation_path=relation_embedding_path,
    entity_dict=entity_dict, 
    relation_dict=relation_dict, 
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
    num_hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    w_matrix=w_matrix,
    bn_list=bn_list)



elif args.mode == 'eval':
    eval(data_path = test_data_path,
    entity_path=entity_embedding_path, 
    relation_path=relation_embedding_path, 
    entity_dict=entity_dict, 
    relation_dict=relation_dict,
    model_path='../../checkpoints/MetaQA/best_score_model.pt',
    train_data=data_path,
    gpu=args.gpu,
    hidden_dim=args.hidden_dim,
    relation_dim=args.relation_dim,
    embedding_dim=args.embedding_dim)
