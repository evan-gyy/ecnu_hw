import os

class Config(object):
    def __init__(self):
        root_path = 'datasets'
        self._get_path(root_path)
        self.batch_size = 8
        self.max_length = 512
        self.epoch = 50
        self.lr = 1e-1
        
        self.patience = 9
        self.training_criteria = 'micro_f1'

        self.gat_layers = 2
        self.hidden_size = 768

        self.semeval_class = 6
        self.class_nums = None

        self.seed = 2020

        self.pool_type = 'avg'
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        self.semeval_ckpt = 'checkpoint/semeval.pth.tar'


    def _get_path(self, root_path):
        self.root_path = root_path
        # bert base uncase bert\bert-base-uncased
        self.bert_base = os.path.join(root_path, 'bert/bert-base-uncased')
        self.bert_base_case = os.path.join(root_path, 'bert/bert-base-cased')

        # dataset path
        self.semeval_path = 'data/hw/'
        self.semeval_rel2id = os.path.join(root_path, self.semeval_path + 'srel2id.json')
        self.semeval_train = os.path.join(root_path, self.semeval_path + 'train.txt')
        self.semeval_val = os.path.join(root_path, self.semeval_path + 'dev.txt')
        self.semeval_test = os.path.join(root_path, self.semeval_path + 'test.txt')
