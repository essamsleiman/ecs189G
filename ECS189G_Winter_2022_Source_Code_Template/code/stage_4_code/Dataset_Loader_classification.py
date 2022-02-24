'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
import torch
import os
import string

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
        image_t = torch.Tensor(row['text'])
        image = image_t
        # image =  torch.unsqueeze(image_t, dim=0)
        # print("IMG SHAPE1: ", image_t.shape)
        # image =  torch.unsqueeze(image_t, dim=0)

        # [112, 92, 3]
        image = image_t.view(3, 32, 32)
        # print("IMG SHAPE2: ", image.shape)
        return image, row['label']

    def cleanup(self, text):
        # split into words by white space
        # print(text)
        words = text.split()
        words = [word.lower() for word in words]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        # print(stripped)
        return stripped
        # convert to lower case

 

    def get_data(self):
        print("dataset_source_folder_path: ",self.dataset_source_folder_path)
        # do posititve and then negative:
        # iterate over files in
        # that directory
        postitve = self.dataset_source_folder_path + "/pos"

        dataset = []

        # print(postitve)
        for filename in os.listdir(postitve):
            f = os.path.join(postitve, filename)
            print("JAMES: ", filename)
            # checking if it is a file
            if os.path.isfile(f):
                myfile = open(f, 'r')
                # this readlines() function allows us to get the content of the text file
                # print(myfile.readlines())
                # print(myfile.read())
                mytext = self.cleanup(myfile.read())
                dataset.append({'text': mytext, 'label': 0})
                # print(f)
        # print("JAMES DATASET")
        # print(dataset)

        negative = self.dataset_source_folder_path + "/neg"
        # print(negative)
        for filename in os.listdir(negative):
            f = os.path.join(negative, filename)
            # checking if it is a file
            if os.path.isfile(f):
                myfile = open(f, 'r')
                mytext = self.cleanup(myfile.read())
                dataset.append({'text': mytext, 'label': 1})
                # this readlines() function allows us to get the content of the text file
                # print(myfile.readlines())
                # print(f)

        # 0 for postitive, 1 for negative

        return dataset