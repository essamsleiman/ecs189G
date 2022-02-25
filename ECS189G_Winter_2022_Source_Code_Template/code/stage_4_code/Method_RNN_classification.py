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




class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self,  mName, mDescription):
        method.__init__(self, mName, mDescription)
        super(Method_RNN, self).__init__()
        nn.Module.__init__(self)

        hidden_dim = 128
        input_size = 2
        n_layers = 2
        output_size = 2
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    # def forward(self, input, hidden):
    #     combined = torch.cat((input, hidden), 1)
    #     hidden = self.i2h(combined)
    #     output = self.i2o(combined)
    #     output = self.softmax(output)
    #     return output, hidden

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        print(x)
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

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
            