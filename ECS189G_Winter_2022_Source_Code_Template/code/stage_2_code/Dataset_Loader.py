'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import csv
import torch

class Dataset_Loader(dataset):
    #data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.data = None
    
    def __len__(self):
        return len(self.data)
        # return 60000
        
    def __getitem__(self, idx):
        row = self.data[idx]
        return row[1:], row[0]

    def get_data(self):
        print("dataset_source_folder_path: ",self.dataset_source_folder_path)
        csvfile = open(self.dataset_source_folder_path + self.dataset_source_file_name, newline='')
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        returnSpamReader = []
        for row in spamreader:
            elements = [int(i) for i in row[0].split(',')]
            returnSpamReader.append(elements)
        return torch.Tensor(returnSpamReader)