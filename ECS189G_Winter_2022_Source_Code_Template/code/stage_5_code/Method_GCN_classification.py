'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from re import I
from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random



class Method_GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 30
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = .01

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self,  mName, mDescription):
        method.__init__(self, mName, mDescription)
        super(Method_GCN, self).__init__()
        nn.Module.__init__(self)
        # cora set
        # self.conv1 = GCNConv(1433, 16)
        # self.conv2 = GCNConv(16, 7)     

        # citeseer set
        # self.conv1 = GCNConv(3703, 16)
        # self.conv2 = GCNConv(16, 6)    
        
        # pubmed set
        self.conv1 = GCNConv(500, 16)
        self.conv2 = GCNConv(16, 3)    


    def forward(self, data):
        print("data: ", data)
        x, edge_index = data.x, data.edge_index

        # print("X b: ", x.size())
        # print("edge_index b: ", edge_index.size())
        x = self.conv1(x, edge_index)
        # print("X a: ", x.size())
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    
    def train(self, data, idx_train, idx_val):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # shuffle = True?
        # train_dataloader = DataLoader([X, y], batch_size=64)
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        # loader = DataLoader(data, shuffle=True)
        train_loss_total_list = []
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            print("epoch: ", epoch)
            y_pred_total = []
            y_true_total = []
            train_loss_epoch_list = []
            # for batch in loader:


            optimizer.zero_grad()
            print("data: ", data)
            y_pred  = self.forward(data)
            #print("y_pred: ", y_pred)
            # y_pred_total.extend(y_pred.max(1)[1])
            print("b y_pred: ", y_pred.size())
            # y_pred = y_pred.max(1)[1]

            # t = torch.rand(4, 2, 3, 3)
            idx = torch.randperm(idx_train.shape[0])
            idx_train = idx_train[idx].view(idx_train.size())

            y_true = torch.LongTensor(np.array(data.y))
            
            y_pred = torch.index_select(y_pred, 0, idx_train)
            print("b y_true: ", y_true.size())
            y_true = torch.index_select(y_true, 0, idx_train)
            print("1 y_pred: ", y_pred.size())
            print("1 y_true: ", y_true.size())



            # y_pred = y_pred[idx_train]
            # idx_train = list(idx_train.cpu().detach().numpy())
            # print("idx_train: ", len(idx_train))
            # random.shuffle(idx_train)
            
            # print("index_shuffle: ", idx_train)
            
            # print("y_pred 1: ", y_pred.size())
            # y_pred = y_pred.cpu().numpy()
            # y_true = y_true.cpu().numpy()
            # print("y_pred 2: ", len(y_pred))
            # idx = torch.randperm(idx_train.shape[0])
            # idx_train = idx_train[idx].view(idx_train.size())
            # print("index_shuffle len: ", len(idx_train))
            # y_pred = y_pred[idx_train]
            # print("y_pred 101: ", y_pred.size())

            # y_pred = torch.from_numpy(y_pred[idx_train])
            # print("y_pred 3: ", y_pred)
            # y_true = torch.from_numpy(y_true[idx_train])
            # print("y_pred: ", y_pred.size())
            # print("y_true: ", y_true.size())
            y_true_total.extend(y_true)
            y_pred_total.extend(y_pred.max(1)[1])

            train_loss = loss_function(y_pred, y_true)
            train_loss_epoch_list.append(train_loss.item())

            train_loss.backward()
            optimizer.step()



            avg_loss = sum(train_loss_epoch_list) / len(train_loss_epoch_list)
            train_loss_total_list.append(avg_loss)
            accuracy_evaluator.data = {'true_y': y_true_total, 'pred_y': y_pred_total}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', avg_loss)
        print("TOTAL LOSS: ", train_loss_total_list)
    def test(self, data, idx_test):
        # do the testing, and result the result
        actual_y = []
        total_pred = []
        # for i, (X,y) in enumerate(data_loader):
        y_true = torch.LongTensor(np.array(data.y))
        y_pred = self.forward(data)

        y_pred = torch.index_select(y_pred, 0, idx_test)
        y_true = torch.index_select(y_true, 0, idx_test)


        total_pred.extend(y_pred.max(1)[1].tolist())

        actual_y.extend(y_true.tolist())
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return total_pred, actual_y
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'], self.data['idx_train'], self.data['idx_val'])
        print('--start testing...')
        pred_y, actual_y = self.test(self.data['test'],  self.data['idx_test'])
        print("pred_y: ", pred_y)
        print("actual_y: ", actual_y)
        return {'pred_y': pred_y, 'true_y': actual_y}