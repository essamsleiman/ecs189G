'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
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
        # print('james row:', row[0][0])
        return row[0], row[1]

    def get_data(self):
        print("dataset_source_folder_path: ",self.dataset_source_folder_path)
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        open_data = pickle.load(f)
        dataset = []
        if (self.dataset_type == 'train'):
            dataset = open_data['train']
        else:
            dataset = open_data['test']
        returnSpamReader = []
        for pair in dataset:
            elements = []
            # plt.imshow(pair['image'], cmap="Greys")
            # plt.show()
            # print(pair['label'])
            # print(pair['image'])
            elements.append(torch.Tensor(pair['image']))
            elements.append(pair['label'])
            # print(elements[0])
            # elements.append(pair['label'])
            returnSpamReader.append(elements)
        return returnSpamReader