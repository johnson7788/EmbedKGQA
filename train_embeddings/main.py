from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from tqdm import tqdm
import os

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., outfile='tucker.model', valid_steps=1, loss_type='BCE', do_batch_norm=1,
                 dataset='', model='Rotat3', l3_reg = 0.0, load_from = ''):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.outfile = outfile
        self.valid_steps = valid_steps
        self.model = model
        self.l3_reg = l3_reg
        self.loss_type = loss_type
        self.load_from = load_from
        if do_batch_norm == 1:
            do_batch_norm = True
        else:
            do_batch_norm = False
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "model": model, "loss_type": loss_type,
                       "do_batch_norm": do_batch_norm, "l3_reg": l3_reg}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            # triple： (214577, 387, 1886604) 转成 （6357，440）：[244510]
            er_vocab[(triple[0], triple[1])].append(triple[2]) # （6357，440）：[244510]
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        #batch 128 ：[(1547903, 866),(1356442, 644).....]
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        #targets:tensor(128,1886683)向量([[0., 0., 0.,  ..., 0., 0., 0.],............
        targets = torch.zeros([len(batch), len(d.entities)], dtype=torch.float32)
        if self.cuda:
            targets = targets.cuda()

        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets

    def evaluate(self, model, data):
        model.eval()
        hits = [[] for _ in range(10)]
        ranks = []
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(test_data_idxs)

        print("Number of data points: %d" % len(test_data_idxs))
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch = np.array(test_data_idxs[i: i+self.batch_size])
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            # following lines commented means RAW evaluation (not filtered)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        hitat10 = np.mean(hits[9])
        hitat3 = np.mean(hits[2])
        hitat1 = np.mean(hits[0])
        meanrank = np.mean(ranks)
        mrr = np.mean(1./np.array(ranks))
        print('Hits @10: {0}'.format(hitat10))
        print('Hits @3: {0}'.format(hitat3))
        print('Hits @1: {0}'.format(hitat1))
        print('Mean rank: {0}'.format(meanrank))
        print('Mean reciprocal rank: {0}'.format(mrr))
        return [mrr, meanrank, hitat10, hitat3, hitat1]

    def write_embedding_files(self, model):
        model.eval()
        model_folder = "../kg_embeddings/%s/" % self.dataset
        data_folder = "../data/%s/" % self.dataset
        embedding_type = self.model
        if os.path.exists(model_folder) == False:
            os.mkdir(model_folder)
        R_numpy = model.R.weight.data.cpu().numpy()
        E_numpy = model.E.weight.data.cpu().numpy()
        bn_list = []
        for bn in [model.bn0, model.bn1, model.bn2]:
            bn_weight = bn.weight.data.cpu().numpy()
            bn_bias = bn.bias.data.cpu().numpy()
            bn_running_mean = bn.running_mean.data.cpu().numpy()
            bn_running_var = bn.running_var.data.cpu().numpy()
            bn_numpy = {}
            bn_numpy['weight'] = bn_weight
            bn_numpy['bias'] = bn_bias
            bn_numpy['running_mean'] = bn_running_mean
            bn_numpy['running_var'] = bn_running_var
            bn_list.append(bn_numpy)
            
        if embedding_type == 'TuckER':
            W_numpy = model.W.detach().cpu().numpy()
            
        np.save(model_folder +'/E.npy', E_numpy)
        np.save(model_folder +'/R.npy', R_numpy)
        for i, bn in enumerate(bn_list):
            np.save(model_folder + '/bn' + str(i) + '.npy', bn)

        if embedding_type == 'TuckER':
            np.save(model_folder +'/W.npy', W_numpy)

        f = open(data_folder + '/entities.dict', 'r')
        f2 = open(model_folder + '/entities.dict', 'w')
        ents = {}
        idx2ent = {}
        for line in f:
            line = line.rstrip().split('\t')
            name = line[0]
            id = int(line[1])
            ents[name] = id
            idx2ent[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()
        f = open(data_folder + '/relations.dict', 'r')
        f2 = open(model_folder + '/relations.dict', 'w')
        rels = {}
        idx2rel = {}
        for line in f:
            line = line.strip().split('\t')
            name = line[0]
            id = int(line[1])
            rels[name] = id
            idx2rel[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()


    def train_and_eval(self):
        # 设置2个线程
        torch.set_num_threads(2)
        # 记录最佳验证
        best_valid = [0, 0, 0, 0, 0]
        # 记录最佳测试
        best_test = [0, 0, 0, 0, 0]
        # 获取实体的下标43234个 188万个实体
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        # 获取关系的下标18个  1000多个关系
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        # 打开实体字典，'../data/fbwq_full/entities.dict' 写入实体值和id
        f = open('../data/' + self.dataset +'/entities.dict', 'w')
        for key, value in self.entity_idxs.items():
            f.write(key + '\t' + str(value) +'\n')
        f.close()
        #打开关系字典，../data/fbwq_full/relations.dict'写入关系和id
        f = open('../data/' + self.dataset + '/relations.dict', 'w')
        for key, value in self.relation_idxs.items():
            f.write(key + '\t' + str(value) +'\n')
        f.close()
        #train_data 训练实体三元组数据转成id向量表示
        train_data_idxs = self.get_data_idxs(d.train_data)
        # 多少个训练数据267164
        print("训练的三元组数据个数是: %d" % len(train_data_idxs))
        # 多少个实体和关系
        print('实体个数: %d' % len(self.entity_idxs))
        print('关系个数: %d' % len(self.relation_idxs))
        # 选择模型 实体E：Embedding(1886683, 400, padding_idx=0)，关系R：Embedding(1144, 400, padding_idx=0)
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.init()
        if self.load_from != '':
            fname = self.load_from
            checkpoint = torch.load(fname)
            model.load_state_dict(checkpoint)
        if self.cuda:
            model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)
        # 将训练的三元组格式， (214577, 387, 1886604) 转成 （6357，440）：[244510]
        er_vocab = self.get_er_vocab(train_data_idxs)
        # er_vocab_pairs：[（6357，440）,.....] 由头实体和关系的编码id组成
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")

        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            if it%100 == 0:
                print('Epoch', it, ' Epoch time', time.time()-start_train, ' Loss:', np.mean(losses))
            model.eval()
            
            with torch.no_grad():
                if it % self.valid_steps == 0:
                    start_test = time.time()
                    print("Validation:")
                    valid = self.evaluate(model, d.valid_data)
                    print("Test:")
                    test = self.evaluate(model, d.test_data)
                    valid_mrr = valid[0]
                    test_mrr = test[0]
                    if valid_mrr >= best_valid[0]:
                        best_valid = valid
                        best_test = test
                        # print('Validation MRR increased.')
                        print('验证的MRR增加.')
                        print('保存模型...')
                        self.write_embedding_files(model)
                        print('模型已保存!')
                    
                    print('最好评估结果:', best_valid)
                    print('最好测试结果:', best_test)
                    print('数据集:', self.dataset)
                    print('模型:', self.model)

                    print(time.time()-start_test)
                    print('Learning rate %f | Decay %f | Dim %d | Input drop %f | Hidden drop 2 %f | LS %f | Batch size %d | Loss type %s | L3 reg %f' % 
                        (self.learning_rate, self.decay_rate, self.ent_vec_dim, self.kwargs["input_dropout"], 
                         self.kwargs["hidden_dropout2"], self.label_smoothing, self.batch_size,
                         self.loss_type, self.l3_reg))        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="waishen", nargs="?",
                    help="使用哪个数据集: waishen,fbwq_full, FB15k-237, MetaQA or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="迭代的次数.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="批次大小.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="学习率")#
    parser.add_argument("--model", type=str, default='ComplEx', nargs="?",
                    help="使用训练的模型:Model.ComplEx，Rotat3，SimplE，DistMult，RESCAL，TuckER")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="衰减率.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="实体嵌入的维度.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="关系嵌入的维度.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="是否使用显卡.True,False ")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="输入层的噪音.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="在第一个隐藏层后的噪音.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="在第二个隐藏层后的噪音.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="标签平滑的数量.")
    parser.add_argument("--outfile", type=str, default='tucker.model', nargs="?",
                    help="输出文件保存名字")
    parser.add_argument("--valid_steps", type=int, default=1, nargs="?",
                    help="在你验证之前的epoch")
    parser.add_argument("--loss_type", type=str, default='BCE', nargs="?",
                    help="损失函数")
    parser.add_argument("--do_batch_norm", type=int, default=1, nargs="?",
                    help="是否使用批次归一化 (0, 1)")
    parser.add_argument("--l3_reg", type=float, default=0.0, nargs="?",
                    help="l3 reg超参数")
    parser.add_argument("--load_from", type=str, default='', nargs="?",
                    help="从状态描述中加载")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "../data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20 #确保每个epoch的批数据是一样的
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) ##
    # data类，处理实体和关系的表示形式
    d = Data(data_dir=data_dir, reverse=True)
    #初始化所有的参数
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, outfile=args.outfile,
                            valid_steps=args.valid_steps, loss_type=args.loss_type, do_batch_norm=args.do_batch_norm,
                            dataset=args.dataset, model=args.model, l3_reg=args.l3_reg, load_from=args.load_from)

    experiment.train_and_eval()
                

