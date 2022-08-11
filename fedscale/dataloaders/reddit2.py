import json
import logging
import os
import warnings

import numpy as np
import json



class reddit2():
    # classes = []
    # MAX_SEQ_LEN = 20000
    VOCAB_SIZE = 0
    my_vocab = {}

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data



    def __init__(self, root, train=True):
        self.train = train  # training set or test set
        self.root = root

        # self.train_file = 'train'
        # self.test_file = 'test'
        # self.train = train

        # self.vocab_tokens_size = 10000
        # self.vocab_tags_size = 500

        # # load data and targets
        # self.raw_data, self.dict = self.load_file(self.root, self.train)

        # if not self.train:
        #     self.raw_data = self.raw_data[:100000]
        # else:
        #     self.raw_data = self.raw_data[:10000000]

        # # we can't enumerate the raw data, thus generating artificial data to cheat the divide_data_loader
        # self.data = [-1, len(self.dict)]
        # self.targets = [-1, len(self.dict)]

    def __getitem__(self, index):
        """
        Args:xx
            index (int): Index
        Returns:
            tuple: (text, tags)
        """

        # Lookup tensor

        # tokens = self.raw_data[index]
        # tokens = torch.tensor(tokens, dtype=torch.long)
        # tokens = F.one_hot(tokens, self.vocab_tokens_size).float()
        # tokens = tokens.mean(0)

        # tags = torch.tensor(tags, dtype=torch.long)
        # tags = F.one_hot(tags, self.vocab_tags_size).float()
        # tags = tags.sum(0)

        return tokens

    def __mapping_dict__(self):
        return self.dict

    def __len__(self):
        return len(self.raw_data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.my_vocab)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))














    def word_index_train(self,word):
        if word in self.my_vocab:
            return self.my_vocab[word]
        else:
            self.my_vocab[word] = self.VOCAB_SIZE
            self.VOCAB_SIZE += 1
            return self.VOCAB_SIZE - 1

    def word_index_test(self, word):
        if word in self.my_vocab:
            return self.my_vocab[word]
        else:
            return 3

    def trans_train(self,x):
        return np.array([[self.word_index_train(word) for word in x_item] for x_item in x], dtype = np.int64)

    def trans_test(self,x):
        return np.array([[self.word_index_test(word) for word in x_item] for x_item in x], dtype = np.int64)



def read_data(train_data_dir, test_data_dir):
    global VOCAB_SIZE
    VOCAB_SIZE = 4
    global my_vocab
    my_vocab = {'<PAD>' : 0, '<BOS>' : 1, '<EOS>' : 2, '<OOV>' : 3}

    clients = []
    groups = []
    train_data,test_data = {},{}
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
    for f in train_files:
        client_name = f.split('.')[0]
        clients.append(client_name)
        train_data[client_name] = {'x':np.array([], dtype = np.int64).reshape(-1, 10), 'y':np.array([], dtype = np.int64).reshape(-1, 10)}

        file_path = os.path.join(train_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]

        for x in my_data:
            train_data[client_name]['y'] = np.vstack((train_data[client_name]['y'], trans_train(x[0]['target_tokens'])))
            train_data[client_name]['x'] = np.vstack((train_data[client_name]['x'], trans_train(x[1])))
            # print(client_name, train_data[client_name]['y'].shape, train_data[client_name]['x'].shape)

    
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
    for f in test_files:
        client_name = f.split('.')[0]
        test_data[client_name] = {'x':np.array([], dtype = np.int64).reshape(-1, 10), 'y':np.array([], dtype = np.int64).reshape(-1, 10)}

        file_path = os.path.join(test_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]
        
        for x in my_data:
            test_data[client_name]['y'] = np.vstack((test_data[client_name]['y'], trans_test(x[0]['target_tokens'])))
            test_data[client_name]['x'] = np.vstack((test_data[client_name]['x'], trans_test(x[1])))


    return clients, groups, train_data, test_data


def load_partition_data_reddit(batch_size,
                              train_path="../data/reddit/train/",
                              test_path="../data/reddit/test/"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    global VOCAB_SIZE
    class_num = VOCAB_SIZE
    # to update the number of classes

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

