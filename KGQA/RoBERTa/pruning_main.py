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
from pruning_dataloader import DatasetPruning, DataLoaderPruning
from pruning_model import PruningModel
from torch.optim.lr_scheduler import ExponentialLR
import networkx as nx
from collections import defaultdict



parser = argparse.ArgumentParser()


parser.add_argument('--ls', type=float, default=0.1,help='')
parser.add_argument('--validate_every', type=int, default=5,help='')

parser.add_argument('--batch_size', type=int, default=16,help='批次大小')
parser.add_argument('--decay', type=float, default=1.0,help='')
parser.add_argument('--shuffle_data', type=bool, default=True,help='是否会打乱输入数据的顺序')
parser.add_argument('--num_workers', type=int, default=0,help='设置工作的线程，多进程debug会卡死,所以设置为0')
parser.add_argument('--lr', type=float, default=0.0002,help='学习率')
parser.add_argument('--nb_epochs', type=int, default=90,help='epoch值，训练的次数')
parser.add_argument('--gpu', type=int, default=0,help='当有多块显卡是，选择第几块，默认0，选择第一块')
parser.add_argument('--use_cuda', type=bool, default=True,help='是否使用GPU，显卡')
parser.add_argument('--patience', type=int, default=5,help='训练停止是的参数')

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()

def printRelationText(rel_ids, idx2rel):
    rel_text = []
    for r in rel_ids:
        if r not in idx2rel:
            r = r.item()
        rel_text.append(idx2rel[r])
    print(rel_text)

def validate_v2(model, device, train_dataset, rel2idx, idx2rel):

    model.eval()

    data = process_data_file('pruning_test.txt', rel2idx, idx2rel)

    num_correct = 0
    count = 0
    for i in tqdm(range(len(data))):
        # try:
        d = data[i]

        question = d[0]

        question_tokenized, attention_mask = train_dataset.tokenize_question(question)

        question_tokenized = question_tokenized.to(device)

        attention_mask = attention_mask.to(device)

        rel_id_list = d[1]
        #得出预测值（18478，）
        scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        #top2：tensor([6058], device='cuda:0')
        top2 = torch.topk(scores, 1)
        #获得三元组答案
        top2 = top2[1]

        isCorrect = False

        for x in top2:

            if x in rel_id_list:
                #如果预测的答案正确
                isCorrect = True
        if isCorrect:
            num_correct += 1
        # else:
        #     print(d[2])
        #     printRelationText(top2, idx2rel)
        #     printRelationText(rel_id_list, idx2rel)
        #     count += 1
        #     if count == 10:
        #         exit(0)
        # pred_rel_id = torch.argmax(scores).item()
        # if pred_rel_id in rel_id_list:
        #     num_correct += 1

            
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)

    accuracy = num_correct/len(data)
    return accuracy

def writeToFile(lines, fname):

    f = open(fname, 'w')

    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def process_data_file(fname, rel2idx, idx2rel):

    f = open(fname, 'r')
    data = []
    #line字符串格式：'what is the name of justin bieber brother [m.06w2sn5]	people.sibling_relationship.sibling|people.person.sibling_s
    for line in f:
        #line转成列表格式：['what is the name of justin bieber brother [m.06w2sn5]', 'people.sibling_relationship.sibling|people.person.sibling_s']
        line = line.strip().split('\t')
        #question：'what is the name of justin bieber brother [m.06w2sn5]'
        question = line[0].strip()
        #TODO only work for webqsp. to remove entity from metaqa, use something else
        #把[]中实体从问题中移除 question：'what is the name of justin bieber brother '
        question = question.split('[')[0]
        #答案列表：rel_list：['people.sibling_relationship.sibling', 'people.person.sibling_s']
        rel_list = line[1].split('|')
        #答案列表的id格式 rel_id_list：[7242, 7212]
        rel_id_list = []

        for rel in rel_list:

            rel_id_list.append(rel2idx[rel])
        #data：[('what is the name of justin bieber brother ', [7242, 7212], 'what is the name of justin bieber brother [m.06w2sn5]')]
        data.append((question, rel_id_list, line[0].strip())) #data：('what is the name of justin bieber brother ', [886, 880], 'what is the name of justin bieber brother [m.06w2sn5]')
    return data

def train(batch_size, shuffle, num_workers, nb_epochs, gpu, use_cuda, patience, validate_every, lr, decay, ls):
    # f = open('/scratche/home/apoorv/mod_TuckER/models/ComplEx_fbwq_full/relations.dict', 'r')
    f = open('/home/wac/johnson/kg/EmbedKGQA-master/data/fbwq_full/relations_all.dict', 'r')
    # 获取关系-id的字典格式 1144个
    rel2idx = {}
    # 获取id-关系的字典格式 1144个
    idx2rel = {}
    #line：american_football.football_coach.coaching_history	2
    for line in f:
        #line：['american_football.football_coach.coaching_history', '2']
        line = line.strip().split('\t')
        #id:2
        id = int(line[1])
        #rel字符串:'american_football.football_coach.coaching_history'
        rel = line[0]
        #答案：id 字典 一共18478
        rel2idx[rel] = id
        #di：答案 字典
        idx2rel[id] = rel
    f.close()
    #处理问题数据集3036个问题，问题变成：[('what is the name of justin bieber brother ', [7242, 7212], 'what is the name of justin bieber brother [m.06w2sn5]')]
    data = process_data_file('pruning_train.txt', rel2idx, idx2rel)

    device = torch.device(gpu if use_cuda else "cpu")
    # 下载tokenizer的参数
    dataset = DatasetPruning(data=data, rel2idx = rel2idx, idx2rel = idx2rel)
    # batchsize=16，一个dataloader=3036/16=190=data_loader
    data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #模型初始化
    model = PruningModel(rel2idx, idx2rel, ls)
    # checkpoint_file = "checkpoints/pruning/best_best.pt"
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint)
    # print('loaded from ', checkpoint_file)
    model.to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = ExponentialLR(optimizer, decay)

    optimizer.zero_grad()

    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0

    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()

                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    #batch_size :16  a:list:3(16,64)
                    model.zero_grad()
                    #问答向量
                    question_tokenized = a[0].to(device)
                    #attention_mask向量
                    attention_mask = a[1].to(device)
                    #答案的向量
                    rel_one_hot = a[2].to(device)
                    # 损失
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, rel_one_hot=rel_one_hot)

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

                score = validate_v2(model=model, device=device, train_dataset=dataset,rel2idx=rel2idx, idx2rel = idx2rel)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0

                    best_model = model.state_dict()

                    print("评估准确率为", score)

                    #writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')

                    checkpoint_path = '../../checkpoints/pruning/'
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    checkpoint_file_name = checkpoint_path + "pruning_robert_1hop.pt"
                    print('保存checkpoint到', checkpoint_file_name)
                    #保存这个模型
                    torch.save(model.state_dict(), checkpoint_file_name)

                    # torch.save(model.state_dict(), "checkpoints/pruning/best_mar12_3.pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1

                    print("评估准去率 从 %f 下降到 %f, 还需要 %d 个epoch去验证 "%( best_score, score,patience-no_update))
                elif no_update == patience:
                    print("准去率不在提高，保存模型并退出")
                    torch.save(best_model, checkpoint_path + "best_score_model.pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("epoch已经完成，停止训练并保存模型.")
                    torch.save(best_model, checkpoint_path + "best_score_model.pt")
                    exit()

def data_generator(data, roberta_file, entity2idx):

    question_embeddings = np.load(roberta_file, allow_pickle=True)

    for i in range(len(data)):

        data_sample = data[i]

        head = entity2idx[data_sample[0].strip()]

        question = data_sample[1]

        # encoded_question = question_embedding[question]

        encoded_question = question_embeddings.item().get(question)

        if type(data_sample[2]) is str:

            ans = entity2idx[data_sample[2]]

        else:

            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question), ans, data_sample[1]



train(
    batch_size=args.batch_size,
    shuffle=args.shuffle_data, 
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    gpu=args.gpu, 
    use_cuda=args.use_cuda, 
    patience=args.patience,
    validate_every=args.validate_every,
    lr=args.lr,
    decay=args.decay,
    ls=args.ls)
