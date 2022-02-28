'''
Base SettingModule class for all experiment settings
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc

#-----------------------------------------------------
class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''
    
    setting_name = None
    setting_description = None
    
    trainDataset = None
    testDataset = None
    method = None
    result = None
    evaluate = None

    def __init__(self, dataset, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription
        self.dataset_object = dataset
    
    def prepare(self, trainDataset, testDataset, sMethod, sResult, sEvaluate):
        self.trainDataset = trainDataset
        self.testDataset = testDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('dataset:', self.testDataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', self.evaluate.evaluate_name)

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
