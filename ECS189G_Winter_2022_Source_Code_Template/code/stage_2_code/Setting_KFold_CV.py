'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from json import load
from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        
        fold_count = 0
        score_list = []
        # for train_index in loaded_data['X']:
        fold_count += 1
        print('************ Fold:', fold_count, '************')
        # X_train = np.array(train_data_x)
        # y_train = np.array(train_data_y)

        print('james size test:', len(loaded_data['X']), len(loaded_data['X_test']))

        print('james test:', np.array(loaded_data['X']))

        X_train, X_test = np.array(loaded_data['X']), np.array(loaded_data['X_test'])
        y_train, y_test = np.array(loaded_data['y']), np.array(loaded_data['y_test'])
    
        # run MethodModule
        # self.method.data = {'train': {'X': X_train, 'y': y_train}}
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X_test': X_test, 'y_test': y_test}}
        learned_result = self.method.run()
        
        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = fold_count
        self.result.save()
        
        self.evaluate.data = learned_result
        score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)