'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(17472, 40)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
         # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        # print("OUT DIM: ", out.shape)
        out = self.fc1(out)
        
        return out

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, data_loader):
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
        train_loss_total_list = []
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            print("epoch: ", epoch)
            y_pred_total = []
            y_true_total = []
            train_loss_epoch_list = []
            for i, (X,y) in enumerate(data_loader):
                optimizer.zero_grad()
                y_pred = self.forward(X)
                y_pred_total.extend(y_pred.max(1)[1])

                # convert y to torch.tensor as well
                y_true = torch.LongTensor(np.array(y))
                y_true_total.extend(y_true)
                # calculate the training loss

                train_loss = loss_function(y_pred, y_true)
                train_loss_epoch_list.append(train_loss.item())
                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
            avg_loss = sum(train_loss_epoch_list) / len(train_loss_epoch_list)
            train_loss_total_list.append(avg_loss)
            # if epoch%100 == 0:
            accuracy_evaluator.data = {'true_y': y_true_total, 'pred_y': y_pred_total}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', avg_loss)
        print("TOTAL LOSS: ", train_loss_total_list)
    def test(self, data_loader):
        # do the testing, and result the result
        actual_y = []
        total_pred = []
        for i, (X,y) in enumerate(data_loader):

            y_pred = self.forward(X)
            total_pred.extend(y_pred.max(1)[1].tolist())
            actual_y.extend(y.tolist())
            # convert the probability distributions to the corresponding labels
            # instances will get the labels corresponding to the largest probability
        return total_pred, actual_y
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        print('--start testing...')
        pred_y, actual_y = self.test(self.data['test'])
        print("pred_y: ", pred_y)
        print("actual_y: ", actual_y)
        return {'pred_y': pred_y, 'true_y': actual_y}
            