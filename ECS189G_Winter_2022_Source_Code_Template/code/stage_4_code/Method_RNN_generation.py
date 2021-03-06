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
    max_epoch = 45
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = .001

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, dataset, mName, mDescription):
        method.__init__(self, mName, mDescription)
        super(Method_RNN, self).__init__()
        nn.Module.__init__(self)


        self.lstm_size = 256
        self.embedding_dim = 256
        self.num_layers = 3
        self.dataset = dataset
        dataset.get_data()
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)
    
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def train(self, data_loader):
        # self.train()

        # dataloader = DataLoader(data_loader, batch_size=args.batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        train_loss_total_list = []

        for epoch in range(self.max_epoch):
            train_loss_epoch_list = []
            state_h, state_c = self.init_state(self.dataset.args.sequence_length)

            for batch, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()

                y_pred, (state_h, state_c) = self(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

                train_loss_epoch_list.append(loss.item())

                # print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
            avg_loss = sum(train_loss_epoch_list) / len(train_loss_epoch_list)
            train_loss_total_list.append(avg_loss)
            print({ 'epoch': epoch, 'loss': avg_loss })
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
    
    def predict(self, text, next_words=25):
        # self.eval()

        words = text.split(' ')
        state_h, state_c = self.init_state(len(words))
        # print("james: ", self.dataset.uniq_words)

        for i in range(0, next_words):
            try:
                x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
                y_pred, (state_h, state_c) = self(x, (state_h, state_c))

                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
                word_index = np.random.choice(len(last_word_logits), p=p)
                words.append(self.dataset.index_to_word[word_index])
            except:
                return ['invalid input']

        return words
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'], )
        while True:
            input_item = input("Enter a start to a joke: ")
            words = self.predict(input_item)
            res_str = ""
            for word in words:
                res_str = res_str + " " + word
            print(res_str)