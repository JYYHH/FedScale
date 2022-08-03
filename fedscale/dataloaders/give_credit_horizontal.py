from __future__ import print_function

import csv
import os
import os.path
import warnings
import numpy as np
import pandas as pd
import torch


class GCH():
    """
    This class will read in the whole specific fate dataset, in a numpy array
    """
    classes = []

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

    def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False):

        self.data_file = dataset  # 'train', 'test', 'validation'
        self.data_files = [self.data_file + '/give_credit_homo_guest.csv', 
                           self.data_file + '/give_credit_homo_host.csv',
                           self.data_file + '/give_credit_homo_test.csv'
                          ]
        self.root = root # $FEDSCALE_HOME/benchmark/dataset/csv_data/give_credit_horizontal
        # load data and targets
        self.data, self.targets = self.load_file()
        self.data, self.targets = torch.tensor(self.data), torch.tensor(self.targets)
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}
        self.dvd_num = 0

    def __getitem__(self, index):
        # return the data and target of the row 'index'
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_file(self):
        # load meta file to get labels
        if self.data_file == "test":
            path = os.path.join(self.root, self.data_files[-1])
            Numpy = np.array(pd.read_csv(path))
            datas, labels = Numpy[:, 2:], Numpy[:, 1]
            self.dvd_num = 0
        else:
            path = os.path.join(self.root, self.data_files[0])
            Numpy = np.array(pd.read_csv(path))
            datas, labels = Numpy[:, 2:], Numpy[:, 1].reshape(-1, 1)
            self.dvd_num = labels.shape[0]

            path = os.path.join(self.root, self.data_files[1])
            Numpy = np.array(pd.read_csv(path))
            datas, labels = np.vstack((datas, Numpy[:, 2:])), np.vstack((labels, Numpy[:, 1].reshape(-1, 1)))

        return datas, labels
