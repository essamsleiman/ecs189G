from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN_classification import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('classification', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed'
    data_obj.dataset_type = 'link'
    data_obj.dataset_name = 'pubmed'

    # test_data_obj = Dataset_Loader('classification', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora'
    # test_data_obj.dataset_type = 'node'

    method_obj = Method_GCN('Graph neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('setting train test split', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')


    # ------------------------------------------------------

    # # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # # ------------------------------------------------------