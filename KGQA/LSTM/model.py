import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_

class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, relation_dim, num_entities, pretrained_embeddings, device, entdrop, reldrop, scoredrop, l3_reg, model, ls, w_matrix, bn_list, freeze=True):
        """

        :param embedding_dim:  256： 嵌入的维度, 问题词的嵌入的维度
        :type embedding_dim:
        :param hidden_dim: 200
        :type hidden_dim:
        :param vocab_size: 117， 所有的问题的单词的数量
        :type vocab_size:
        :param relation_dim:  200， 关系的维度
        :type relation_dim:
        :param num_entities:  43234， 实体的数量
        :type num_entities:
        :param pretrained_embeddings: list， 43234， 每个实体的嵌入向量
        :type pretrained_embeddings:
        :param device: cpu， 训练设备
        :type device:
        :param entdrop: 0.0
        :type entdrop:
        :param reldrop: 0.0
        :type reldrop:
        :param scoredrop: 0.0
        :type scoredrop:
        :param l3_reg: 0.0
        :type l3_reg:
        :param model: 'ComplEx'
        :type model:
        :param ls: 0.0， label_smoothing值
        :type ls:
        :param w_matrix:  '../../pretrained_models/embeddings/ComplEx_MetaQA_full/W.npy'
        :type w_matrix:
        :param bn_list:  批归一化的模型参数
        :type bn_list:  dict
        :param freeze: True
        :type freeze: bool
        """
        super(RelationExtractor, self).__init__()
        self.device = device
        self.bn_list = bn_list
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        if self.model == 'DistMult':
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'Rotat3':
            multiplier = 3
            self.getScores = self.Rotat3
        elif self.model == 'TuckER':
            W_torch = torch.from_numpy(np.load(w_matrix))
            self.W = nn.Parameter(
                torch.Tensor(W_torch), 
                requires_grad = True
            )
            # self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)), 
            #                         dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('选用的模型是: ', self.model)
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim
        # 嵌入的维度, 问题词的嵌入的维度
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = 1
        self.bidirectional = True
        
        self.num_entities = num_entities
        self.loss = torch.nn.BCELoss(reduction='sum')

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)

        # LSTM将单词嵌入作为输入，并输出维度为hidden_dim的隐藏状态。
        self.pretrained_embeddings = pretrained_embeddings
        print('冻结预训练的实体Embedding的参数:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 256
        self.mid2 = 256

        self.lin1 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)
        self.lin2 = nn.Linear(self.mid1, self.mid2, bias=False)
        xavier_normal_(self.lin1.weight.data)   # 使用xavier方式初始化权重
        xavier_normal_(self.lin2.weight.data)
        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)
        self.hidden2rel_base = nn.Linear(hidden_dim * 2, self.relation_dim)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)
        # bn_list:bn_list
        #  0 = {dict: 4} {'weight': tensor([0.9646, 0.9430]), 'bias': tensor([ 0.1148, -0.0995]), 'running_mean': array([ 0.046767  , -0.05102218], dtype=float32), 'running_var': array([0.00692407, 0.00702443], dtype=float32)}
        #  1 = {dict: 4} {'weight': array([1., 1.], dtype=float32), 'bias': array([0., 0.], dtype=float32), 'running_mean': array([0., 0.], dtype=float32), 'running_var': array([1., 1.], dtype=float32)}
        #  2 = {dict: 4} {'weight': array([0.71846   , 0.68144286], dtype=float32), 'bias': array([-0.5668318,  0.5821315], dtype=float32), 'running_mean': array([ 0.00204753, -0.00344366], dtype=float32), 'running_var': array([0.02376127, 0.023404  ], dtype=float32)}
        for i in range(3):
            for key, value in self.bn_list[i].items():
                self.bn_list[i][key] = torch.Tensor(value).to(device)

        # 批归一化的模型参数
        self.bn0.weight.data = self.bn_list[0]['weight']
        self.bn0.bias.data = self.bn_list[0]['bias']
        self.bn0.running_mean.data = self.bn_list[0]['running_mean']
        self.bn0.running_var.data = self.bn_list[0]['running_var']

        self.bn2.weight.data = self.bn_list[2]['weight']
        self.bn2.bias.data = self.bn_list[2]['bias']
        self.bn2.running_mean.data = self.bn_list[2]['running_mean']
        self.bn2.running_var.data = self.bn_list[2]['running_var']

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)
        

    def applyNonLinear(self, outputs):
        # [batch_size, hidden_dim] --> [batch_size, 256]
        outputs = self.lin1(outputs)  #Linear(in_features=400, out_features=256, bias=False)
        outputs = F.relu(outputs)
        # [batch_size, 256] --> [batch_size, 256]
        outputs = self.lin2(outputs)  #Linear(in_features=256, out_features=256, bias=False)
        outputs = F.relu(outputs)
        # [batch_size, 256] --> [batch_size, rel_hidden] [1024, 400]
        outputs = self.hidden2rel(outputs)   #Linear(in_features=256, out_features=400, bias=True)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def TuckER(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        x = head.view(-1, 1, head.size(1))

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, head.size(1), head.size(1))
        W_mat = self.rel_dropout(W_mat)
        x = torch.bmm(x, W_mat) 
        x = x.view(-1, head.size(1)) 
        x = self.bn2(x)
        x = self.score_dropout(x)

        x = torch.mm(x, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def RESCAL(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        ent_dim = head.size(1)
        head = head.view(-1, 1, ent_dim)
        relation = relation.view(-1, ent_dim, ent_dim)
        relation = self.rel_dropout(relation)
        x = torch.bmm(head, relation) 
        x = x.view(-1, ent_dim)  
        x = self.bn2(x)
        x = self.score_dropout(x)
        x = torch.mm(x, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    
    def SimplE(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = torch.mm(s, self.embedding.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx(self, head, relation):
        """

        :param head: torch.Size([1024, 400])，  头实体的嵌入向量
        :type head:
        :param relation:  torch.Size([1024, 400])， 句子的高阶特征作为关系的表示
        :type relation:
        :return:
        :rtype:
        """
        # [1024, 400]--> tuple,2个元素，把head的按照维度1进程拆分成2个矩阵, 每个的形状是torch.Size([1024, 200])
        head_chunk = torch.chunk(head, 2, dim=1)
        # 拼接到一起,  2个[1024, 400]的矩阵在维度1上拼接后，得到 torch.Size([1024, 2, 200])
        head = torch.stack(list(head_chunk), dim=1)
        # 进行一次batch_normalization
        head = self.bn0(head)
        # 进行一次dropout
        head = self.ent_dropout(head)
        #
        relation = self.rel_dropout(relation)
        #交换batch_size的维度:  torch.Size([1024, 2, 200]) -- > torch.Size([2, 1024, 200])
        head = head.permute(1, 0, 2)
        # 然后拆成2部分， re_head 和im_head : torch.Size([1024, 200])
        re_head = head[0]
        im_head = head[1]
        # 关系的和头实体的一样进行处理, re_relation, im_relation: torch.Size([1024, 200])
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        # 对关系和实体分词2部分进行交互
        # re_score: torch.Size([1024, 200])   << re_head: [1024, 200] * re_relation: : [1024, 200] --> torch.Size([1024, 200]), im_head: [1024, 200] * im_relation: : [1024, 200] --> torch.Size([1024, 200])
        re_score = re_head * re_relation - im_head * im_relation
        #  im_score: torch.Size([1024, 200])
        im_score = re_head * im_relation + im_head * re_relation
        # 2个[1024, 200]，在维度1上拼接， torch.Size([1024, 2, 200])
        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)  # torch.Size([1024, 2, 200])
        score = self.score_dropout(score)
        #和batch_size: torch.Size([2, 1024, 200])
        score = score.permute(1, 0, 2)
        # re_score： torch.Size([1024, 200])， im_score: torch.Size([1024, 200])
        re_score = score[0]
        im_score = score[1]
        #
        # 对实体嵌入也进行拆分， re_tail:im_tail: torch.Size([43234, 200])
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)
        # 矩阵乘法  torch.mm(re_score, re_tail.transpose(1,0))--> torch.Size([1024, 43234]), 最终分数： torch.Size([1024, 43234])
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # 进行一次sigmoid, [batch_size, entities_num]
        pred = torch.sigmoid(score)
        return pred

    def Rotat3(self, head, relation):
        pi = 3.14159265358979323846
        relation = F.hardtanh(relation) * pi
        r = torch.stack(list(torch.chunk(relation, 3, dim=1)), dim=1)
        h = torch.stack(list(torch.chunk(head, 3, dim=1)), dim=1)
        h = self.bn0(h)
        h = self.ent_dropout(h)
        r = self.rel_dropout(r)
        
        r = r.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        x = h[0]
        y = h[1]
        z = h[2]

        # need to rotate h by r
        # r contains values in radians

        for i in range(len(r)):
            sin_r = torch.sin(r[i])
            cos_r = torch.cos(r[i])
            if i == 0:
                x_n = x.clone()
                y_n = y * cos_r - z * sin_r
                z_n = y * sin_r + z * cos_r
            elif i == 1:
                x_n = x * cos_r - y * sin_r
                y_n = x * sin_r + y * cos_r
                z_n = z.clone()
            elif i == 2:
                x_n = z * sin_r + x * cos_r
                y_n = y.clone()
                z_n = z * cos_r - x * sin_r

            x = x_n
            y = y_n
            z = z_n

        s = torch.stack([x, y, z], dim=1)        
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = s.permute(1, 0, 2)
        s = torch.cat([s[0], s[1], s[2]], dim = 1)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    
    def forward(self, sentence, p_head, p_tail, question_len):
        """
        前向传播
        :param sentence:  问题的向量，torch.Size([1024, 11])  [batch_size, batch_max_seq_len]
        :type sentence:
        :param p_head:  问题中头实体的id, [batch_size]
        :type p_head:
        :param p_tail:  torch.Size([1024, 43234]), 答案尾实体的向量[batch_size, num_entities]
        :type p_tail:
        :param question_len: 问题的长度：[batch_size]
        :type question_len:
        :return:
        :rtype:
        """
        # torch.Size([1024, 11]) -->torch.Size([1024, 11, 256]),  [batch_size, batch_max_seq_len]-->  [batch_size, batch_max_seq_len, embedding_dim]
        embeds = self.word_embeddings(sentence)
        # 将一个填充过的变长序列压紧, question_len,问题的长度：[batch_size], embeds, [batch_size, batch_max_seq_len, embedding_dim]
        packed_output = pack_padded_sequence(embeds, question_len.cpu(), batch_first=True)
        #  hidden: torch.Size([2, 1024, 200]), cell_state： torch.Size([2, 1024, 200]), [bidirectional, batch_first, hidden_embedding]
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        # ???
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
        # 拼接双向的输出, torch.Size([2, 1024, 200]) -->torch.Size([1024, 400]), [batch_size, hidden_dim*2]
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)

        # outputs = self.drop1(outputs)
        # rel_embedding = self.hidden2rel(outputs)
        # [1024,400] --> [1024,400], 对问题进行非线性处理后，得到一个高阶特征
        rel_embedding = self.applyNonLinear(outputs)
        # 头实体 [1024] -->torch.Size([1024, 400]) ,
        p_head = self.embedding(p_head)
        # 头实体和问题之间进行交互注意力， pred: [batch_size, entities_num]
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:  # 标签平滑策略，
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(pred, actual)
        # reg = -0.001
        # best: reg is 1.0
        # self.l3_reg = 0.002
        # self.gamma1 = 1
        # self.gamma2 = 3
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss
        
    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)
        return rel_embedding

    def get_score_ranked(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)

        head = self.embedding(head).unsqueeze(0)
        score = self.getScores(head, rel_embedding)
        
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2
        




