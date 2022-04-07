import json


class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data #将训练数据，有效数据，测试数据放在一起
        self.entities = self.get_entities(self.data)#将所有的实体放在一起
        self.train_relations = self.get_relations(self.train_data)#
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False,isjson=True):
        '''
        加载数据
        '''
        if isjson:
            file_name = "%s%s.json" % (data_dir, data_type)
            with open(file_name,"r") as f:
                data = json.load(f)
            if reverse:  # reverse 为true
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]  # 头实体和尾实体调换位置,并加入到原data数据里，data数量 133582*2 = 267164个，格式不变
            # strip 去掉多余\n和空格
            data = [[x.strip() for x in i] for i in data]
            return data

        else:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                data = f.read().strip().split("\n") #一共133582条数据，格式：['The Prowler\tstrred_actor\yn Keyes','...',...]
                data = [i.split('\t') for i in data] #数据格式变为：[['The Prowler','tstrred_actor','yn Keyes'],[],...]
                if reverse: #reverse 为true
                    data += [[i[2], i[1]+"_reverse", i[0]] for i in data] #头实体和尾实体调换位置,并加入到原data数据里，data数量 133582*2 = 267164个，格式不变
            return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
