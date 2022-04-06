import os

class Data:
    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        """
        加载数据
        :param data_dir:'../data/MetaQA/'
        :type data_dir:
        :param data_type:
        :type data_type:
        :param reverse:  True
        :type reverse:bool
        :return:
        :rtype:
        """
        file_name = "%s%s.txt" % (data_dir, data_type)
        print(f"开始读取数据集文件: {file_name}")
        assert os.path.exists(file_name), f"数据文件{file_name}不存在"
        with open(file_name, "r") as f:
            data = f.read().strip().split("\n")  # 所有数据, 数据条数是133582条
            data = [i.split('\t') for i in data]  # list格式，其中一条数据格式 ['The Prowler', 'starred_actors', 'Evelyn Keyes']
            if reverse:  # True
                reverse_data = [[i[2], i[1]+"_reverse", i[0]] for i in data] # 其中一条数据是： ['Evelyn Keyes', 'starred_actors_reverse', 'The Prowler']
                data += reverse_data  # 扩充数据，数据的格式是, data数据条数变成 133582* 2 = 267164条
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
