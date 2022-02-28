'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from calendar import c
from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt
import torch
import os
import string
from sklearn import preprocessing
import random

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
        # print(row['text'])
        text_t = torch.tensor(row['text'], dtype=torch.long)
        text = text_t

        return text, row['label']
    
    def to_ascii(self, text):
        # labels = ['cat', 'dog', 'mouse', 'elephant', 'pandas']
        # le = preprocessing.LabelEncoder()
        # targets = le.fit_transform(labels)
        # print(targets)
        ascii_values = [ord(character) for character in text]
        return ascii_values

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
        print("POS: ", postitve)
        dataset = []
        essamTestDataset = []

        # print(postitve)
        le = preprocessing.LabelEncoder()
        count_pos = 0
        for filename in os.listdir(postitve):
            if count_pos == 500:
                break
            count_pos+=1

            f = os.path.join(postitve, filename)
           #  print("JAMES: ", filename)
            # checking if it is a file
            if os.path.isfile(f):
                myfile = open(f, 'r')
                mytext = self.cleanup(myfile.read())
                targets = le.fit_transform(mytext)
                
                # print(targets)
                dataset.append({'text': targets, 'label': 0})
                essamTestDataset.extend(targets)

        negative = self.dataset_source_folder_path + "/neg"
        count_neg = 0
        for filename in os.listdir(negative):
            if count_neg == 500:
                break
            count_neg+=1
            f = os.path.join(negative, filename)
            # checking if it is a file
            if os.path.isfile(f):
                myfile = open(f, 'r')
                mytext = self.cleanup(myfile.read())
                # le = preprocessing.LabelEncoder()
                targets = le.fit_transform(mytext)
                dataset.append({'text': targets, 'label': 1})
                essamTestDataset.extend(targets)

        # 0 for postitive, 1 for negative
        # print("LEN DATA: ", dataset)
        # setEssam = set(essamTestDataset)
        # print("SUMA: ", setEssam, len(setEssam), max(setEssam)) this cage us size of vocab list
        # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        # print("OMG: ", le_name_mapping)
        random.shuffle(dataset)
        return dataset