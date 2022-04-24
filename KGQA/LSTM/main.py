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
parser.add_argument('--ls', type=float, default=0.0, help='label_smoothing的值')
parser.add_argument('--validate_every', type=int, default=10, help='每隔多少个epoch进行一次验证')
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--kg_type', type=str, default='full',help="可以选择half，或者full")

parser.add_argument('--mode', type=str, default='eval', help='train 还是eval，不同的模式')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
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
parser.add_argument('--patience', type=int, default=30, help='验证集多少次不更新最好指标后，那么就停止训练')
parser.add_argument('--freeze', type=str2bool, default=True, help="是否冻结预训练的实体Embedding的参数")

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    for line in f:   # 每条数据格式: line: '1\t$9.99\n'
        line = line.strip().split('\t')  # ['1', '$9.99']
        ent_id = int(line[0])  # 获取实体id， int
        ent_name = line[1]     #获取实体名称, $9.99
        e[ent_name] = entities[ent_id]  # 获取实体的嵌入向量, 维度 400, 实体名称：实体向量的字典格式
    f.close()
    # e：43234个，代表实体的个数
    f = open(relation_dict,'r')  #打开关系的嵌入
    for line in f:
        line = line.strip().split('\t')  #['0', 'directed_by']
        rel_id = int(line[0])  #int， 关系的id, eg: 0
        rel_name = line[1]  # 'directed_by'
        r[rel_name] = relations[rel_id]  # 这个关系对应向量
    f.close()   # 关系r, dict, 18个关系
    return e,r


def validate(data_path, device, model, word2idx, entity2idx, model_name):
    model.eval()
    data = process_text_file(data_path)
    answers = []
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    correct_length = 0
    for i in tqdm(range(len(data))):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            tail_test = torch.tensor(ans, dtype=torch.long).to(device)
            predict_index = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)
            # top2 = torch.topk(predict, k=2, largest=True, sorted=True)
            # top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()
            predict = np.where(predict_index==1)[1]
            predict = predict.tolist()
            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            if predict == ans:
                total_correct += 1
                is_correct = 1
            if len(predict) == len(ans):
                correct_length += 1
            q_text = d[-1]
            answers.append(q_text + '\t' + str(predict) + '\t' + str(is_correct))
        except Exception as e:
            error_count += 1
            print(e)
            
    print(f"预测的样本个数: {len(data)}, 预测的实体数量和答案数量相同的是: {correct_length}, 预测Exception数量: {error_count}")
    accuracy = total_correct/len(data)
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


def train(data_path, entity_path, relation_path, entity_dict, relation_dict, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, num_hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, w_matrix, bn_list, valid_data_path=None):
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
    entities = np.load(entity_path)  #实体嵌入【实体个数，嵌入维度】 ,(43234, 400)
    relations = np.load(relation_path)  #关系嵌入 (18, 400)， 【关系种类，嵌入维度】
    # 返回e是实体对应的嵌入，字典格式， r是关系对应的嵌入，字典格式,
    e,r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
    # 实体到id的映射的字典，id到实体映射的字典， 实体的嵌入向量的列表格式
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    # 处理数据，data， list，  是【头实体，问题，答案列表】的格式，如果split是FALSE., 这里是208970条训练数据
    data = process_text_file(data_path, split=False)
    # data = pickle.load(open(data_path, 'rb'))
    # 对问题进行处理，获取单词到id， id到单词的映射字典，最大长度
    word2ix,idx2word, max_len = get_vocab(data)
    # eg: '1'
    hops = str(num_hops)
    # print(idx2word)
    # print(idx2word.keys())
    # eg: cpu
    device = torch.device(gpu if use_cuda else "cpu")
    # 初始化数据集
    dataset = DatasetMetaQA(data=data, word2ix=word2ix, relations=r, entities=e, entity2idx=entity2idx)
    # 初始化dataloader, batch_size: 1024
    data_loader = DataLoaderMetaQA(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 初始化模型， embedding_dim： 256， hidden_dim： 200，vocab_size：117， 一共117个单词，问题的单词数量
    model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2ix), num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, w_matrix = w_matrix, bn_list=bn_list)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = ExponentialLR(optimizer, decay)

    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    for epoch in range(nb_epochs):
        phases = ['valid']
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                if freeze == True:
                    print('如果冻结实体的嵌入参数，那么对应的BN的参数也需要冻结')
                    model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    # i_batch返回4个值： 分别是：问题的向量，维度是[batch_size, batch_max_seq_len],  问题的长度：[batch_size]， 问题中头实体的id, [batch_size],  答案尾实体的向量[batch_size, num_entities]
                    question = a[0].to(device)   #问题的向量，torch.Size([1024, 11])  [batch_size, batch_max_seq_len]
                    sent_len = a[1].to(device)   #问题的长度：[batch_size]
                    positive_head = a[2].to(device)  #问题中头实体的id, [batch_size]
                    positive_tail = a[3].to(device)      #torch.Size([1024, 43234]), 答案尾实体的向量[batch_size, num_entities]
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
                    print('验证集最好的分数是 :', test_score)
                    writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    suffix = ''
                    if freeze == True:
                        suffix = '_frozen'
                    checkpoint_path = '../../checkpoints/MetaQA/'
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    checkpoint_file_name = checkpoint_path +model_name+ '_' + num_hops + suffix + ".pt"
                    print('保持checkpoint到:', checkpoint_file_name)
                    torch.save(model.state_dict(), checkpoint_file_name)
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print(f"达到预设的训练的patience，保存最好的模型，训练退出，模型保存到{checkpoint_path}best_score_model.pt")
                    torch.save(best_model, checkpoint_path+ "best_score_model.pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stopping and saving model.")
                    torch.save(best_model, checkpoint_path +"best_score_model.pt")
                    exit()
                    

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
        question_2 = question[1].split(']')  #问题对应的实体
        head = question_2[0].strip()  #问题对应的实体
        question_2 = question_2[1]  # 问题的后半部分
        question = question_1+'NE'+question_2  #问题变成NE连接， 'what movies are about NE'
        ans = data_line[1].split('|')  #答案变成列表格式:['Top Hat', 'Kitty Foyle', 'The Barkleys of Broadway']
        data_array.append([head, question.strip(), ans])  # 一条数据变成【头实体，问题，答案列表】的格式
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


def do_eval(data_path,entity_path, relation_path,  entity_dict, relation_dict,model_path,train_data,gpu, hidden_dim, relation_dim,embedding_dim):
    """
    参考训练时的验证集结果
    :param data_path:
    :type data_path:
    :param entity_path:
    :type entity_path:
    :param relation_path:
    :type relation_path:
    :param entity_dict:
    :type entity_dict:
    :param relation_dict:
    :type relation_dict:
    :param model_path:
    :type model_path:
    :param train_data:
    :type train_data:
    :param gpu:
    :type gpu:
    :param hidden_dim:
    :type hidden_dim:
    :param relation_dim:
    :type relation_dim:
    :param embedding_dim:
    :type embedding_dim:
    :return:
    :rtype:
    """
    pass


hops = args.hops
if hops in ['1', '2', '3']:
    hops = hops + 'hop'
if args.kg_type == 'half':
    data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '_half.txt'
else:
    data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + '.txt'
print('训练文件是 ', data_path)

hops_without_old = hops.replace('_old', '')
valid_data_path = '../../data/QA_data/MetaQA/qa_dev_' + hops_without_old + '.txt'
test_data_path = '../../data/QA_data/MetaQA/qa_test_' + hops_without_old + '.txt'

model_name = args.model
kg_type = args.kg_type
print('KG的类型是', kg_type)
embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_' + kg_type

entity_embedding_path = embedding_folder + '/E.npy'
relation_embedding_path = embedding_folder + '/R.npy'
entity_dict = embedding_folder + '/entities.dict'
relation_dict = embedding_folder + '/relations.dict'
w_matrix =  embedding_folder + '/W.npy'

bn_list = []

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
    do_eval(data_path = test_data_path,
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
