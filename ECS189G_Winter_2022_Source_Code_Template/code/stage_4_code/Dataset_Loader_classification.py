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
from nltk.stem import PorterStemmer

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
		ps =PorterStemmer()
		for i, w in enumerate(stripped):
			rootWord=ps.stem(w)
			stripped[i] = rootWord
		max_len = 400
		if len(stripped) < 400:
			for i in range(len(stripped), 400):
				stripped.append(-1)
		else:
			stripped = stripped[0:400]
		# print(len(stripped)) 
		return stripped
		# convert to lower case

	def get_data(self):
		print("dataset_source_folder_path: ",self.dataset_source_folder_path)
		train_path = '../../data/stage_4_data/text_classification/train'
		test_path = '../../data/stage_4_data/text_classification/test'

		# do posititve and then negative:
		# iterate over files in
		# that directory
		train_postitve = train_path + "/pos"
		train_neg = train_path + "/neg"
		test_postitve = test_path + "/pos"
		test_neg = test_path + "/neg"

		# print("POS: ", postitve)
		dataset = []
		essamTestDataset = []

		# print(postitve)
		le = preprocessing.LabelEncoder()
		total_text = []
		my_label = []
		c = 0
		dataset_len = 500
		for filename in os.listdir(train_postitve):
			# print("train_postitve")
			# if c == dataset_len:
				# break
			# c+=1
			f = os.path.join(train_postitve, filename)
			if os.path.isfile(f):
				myfile = open(f, 'r')
				mytext = self.cleanup(myfile.read())
				total_text.append(mytext)
				my_label.append(1)
				# targets = le.fit_transform(mytext)
				# print(targets)
				# print(targets)
				# dataset.append({'text': targets, 'label': 0})
				# essamTestDataset.extend(targets)

		# negative = self.dataset_source_folder_path + "/neg"
		c = 0
		for filename in os.listdir(train_neg):
			# print("train_neg")

			# if c == dataset_len:
				# break
			# c+=1
			f = os.path.join(train_neg, filename)
			if os.path.isfile(f):
				myfile = open(f, 'r')
				mytext = self.cleanup(myfile.read())
				total_text.append(mytext)
				my_label.append(0)

		c = 0
		for filename in os.listdir(test_postitve):
			# print("test_postitve")

			if c == dataset_len:
				break
			c+=1
			f = os.path.join(test_postitve, filename)
			if os.path.isfile(f):
				myfile = open(f, 'r')
				mytext = self.cleanup(myfile.read())
				total_text.append(mytext)
				my_label.append(1)

		c = 0
		for filename in os.listdir(test_neg):
			# print("test_neg")

			if c == dataset_len:
				break
			c+=1
			f = os.path.join(test_neg, filename)
			if os.path.isfile(f):
				myfile = open(f, 'r')
				mytext = self.cleanup(myfile.read())
				total_text.append(mytext)
				my_label.append(0)

		fit_transform_input =[]
		for text in total_text:
			fit_transform_input.extend(text)

		fit_transform_revert = []
		targets = le.fit_transform(fit_transform_input)
		print("t len: ", len(set(targets)))
		ptr = 0
		for text in total_text:

			fit_transform_revert.append(targets[ptr:ptr+len(text)])
			ptr = ptr+len(text)

		for i, text in enumerate(fit_transform_revert):

			# f = os.path.join(negative, filename)
			# checking if it is a file
			# if os.path.isfile(f):
				# myfile = open(f, 'r')

				# mytext = self.cleanup(myfile.read())
				# total_text.extend(mytext)

				# le = preprocessing.LabelEncoder()
			# print(i)
			# print(text)
			# print(my_label[i])
			# break
			# print(my_label[i])
			dataset.append({'text': text, 'label': my_label[i]})
				# essamTestDataset.extend(targets)

		# 0 for postitive, 1 for negative
		# print("LEN DATA: ", dataset)
		# setEssam = set(essamTestDataset)
		# print("SUMA: ", setEssam, len(setEssam), max(setEssam)) this cage us size of vocab list
		# le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

		# print("OMG: ", le_name_mapping)
		random.shuffle(dataset)
		return dataset