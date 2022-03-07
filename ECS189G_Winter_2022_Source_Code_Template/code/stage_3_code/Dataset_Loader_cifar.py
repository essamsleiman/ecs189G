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
        image_t = torch.Tensor(row['image'])
        image = image_t
        # image =  torch.unsqueeze(image_t, dim=0)
        # print("IMG SHAPE1: ", image_t.shape)
        # image =  torch.unsqueeze(image_t, dim=0)

        # [112, 92, 3]
        image = image_t.view(3, 32, 32)
        # print("IMG SHAPE2: ", image.shape)
        return image, row['label']

    def get_data(self):
        print("dataset_source_folder_path: ",self.dataset_source_folder_path)
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        open_data = pickle.load(f)
        dataset = []
        if (self.dataset_type == 'train'):
            dataset = open_data['train']
        else:
            dataset = open_data['test']

        # for pair in dataset:
        #     pair['label'] = pair['label'] - 1

        # returnSpamReader = []
        # for pair in dataset:
        #     elements = []
        #     # plt.imshow(pair['image'], cmap="Greys")
        #     # plt.show()
        #     # print(pair['label'])
        #     # print(pair['image'])
        #     elements.append(torch.Tensor(pair['image']))
        #     elements.append(pair['label'])
        #     # print(elements[0])
        #     # elements.append(pair['label'])
        #     returnSpamReader.append(elements)
        return dataset