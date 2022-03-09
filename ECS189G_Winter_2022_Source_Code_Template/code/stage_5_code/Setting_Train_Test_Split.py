'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        data, idx_train, idx_val, idx_test  = self.trainDataset.load()
        self.trainDataset.data = data
        # self.trainDataset.data = [self.trainDataset.data, train_miniSet]
        # self.trainDataset.train_miniSet = train_miniSet
        self.testDataset.data = data

        # train_dataloader = DataLoader(self.trainDataset, batch_size=64)
        # test_dataloader = DataLoader(self.testDataset, batch_size=64)   

        # run MethodModule
        self.method.data = {'train': self.trainDataset.data, 'test': self.trainDataset.data, 'idx_train':idx_train, 'idx_val':idx_val, 'idx_test':idx_test}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
   
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None
        