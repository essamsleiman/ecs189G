from code.stage_4_code.Dataset_Loader_generation import Dataset_Loader
from code.stage_4_code.Method_RNN_generation import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_4_code.Settings_generation import Setting_Train_Test_Split
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', type=int, default=4)
    args = parser.parse_args()
    data_obj = Dataset_Loader(args, 'generation', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation'
    data_obj.dataset_source_file_name = 'data'
    data_obj.dataset_type = 'train'

    # test_data_obj = Dataset_Loader('classification', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/test'
    # test_data_obj.dataset_type = 'test'

    method_obj = Method_RNN(data_obj, 'recurrent neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('setting train test split (generation)', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')


    # ------------------------------------------------------

    # # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # # ------------------------------------------------------
