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
import pandas as pd
from collections import Counter
import re
from spellchecker import SpellChecker
spell = SpellChecker()

class Dataset_Loader(dataset):
    #data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, args, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.data = None
        self.args = args
        self.words = None
        self.uniq_words = None
        self.word_to_index = None


    def check(self, word):
        if word == spell.correction(word) or word in string.punctuation:
            return True
        else:
            return False

    def cleanup(self, text):
        mytext = re.findall(r"[\w']+|[.,!?;]", text)
        words = [word.lower() for word in mytext]
        return words

    def get_data(self):
        train_df = pd.read_csv(self.dataset_source_folder_path + '/' + self.dataset_source_file_name)
        text = train_df['Joke'].str.cat(sep=' ')
        self.words = self.cleanup(text)

        # self.words = text.split(' ')
        self.uniq_words = self.get_uniq_words()
        # print("james: ", self.uniq_words)

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]
        return text.split(' ')
    
    def get_uniq_words(self):
        words_counts = Counter(self.words)
        return sorted(words_counts, key=words_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
        # return len(self.data)
        # return 60000
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.words_indexes[idx:idx+self.args.sequence_length]),
            torch.tensor(self.words_indexes[idx+1:idx+self.args.sequence_length+1]),
        )