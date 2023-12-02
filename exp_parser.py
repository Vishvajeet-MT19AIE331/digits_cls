# Program for selecting the best performing models and training and saving them ###########


# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--clf_name", type= str, help="type of model")
parser.add_argument("--random_state",type=int, help="seeding for random state")
args = parser.parse_args()


num_runs  = 1
shuffle_arg = True
random_state = args.random_state
# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
classifier_param_dict = {}
# 2.1. SVM
gamma_list = [0.001] #[0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1] #[0.1, 1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations
#print('__________________________________________________________________________________________________________________')
#print('TRAINING LOGS:')
#print('The combination of tested parameters for SVM are:')
#print(h_params_combinations)

# 2.2 Decision Tree
max_depth_list = [5]#[5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations
#print()
#print('The combination of tested parameters for Decision Tree are:')
#print(h_params_trees_combinations)
#print('__________________________________________________________________________________________________________________')

# Choose only one of the models based on argument supplied
if args.clf_name == 'tree':
    del classifier_param_dict['svm'] 
else:
    del classifier_param_dict['tree'] 


results = []
test_sizes =  [0.2]
dev_sizes  =  [0.2]
for cur_run_i in range(num_runs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, 
                                            test_size=test_size, dev_size=dev_size, random_state=random_state, shuffle_arg=shuffle_arg)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            binary_preds = {}
            model_preds = {}
            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                y_dev, current_hparams, model_type)        
            
                # loading of model         
                best_model = load(best_model_path) 

                test_acc, test_f1, predicted_y = predict_and_eval(best_model, X_test, y_test)
                train_acc, train_f1, _ = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy
                
                #print('BEST RESULTS:')
                #print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}, test_f1={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc, test_f1))
                #print('Best model:',best_model_path)
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)
                binary_preds[model_type] = y_test == predicted_y
                model_preds[model_type] = predicted_y
                
                file_name = "./results/{}_{}".format(model_type,random_state)+".txt"
                f = open(file_name, 'a')  
                print("test accuracy:{:.5f}, test macro-f1: {:.5f}, model saved at".format( test_acc, test_f1),best_model_path, file=f) 
                f.close()

                print("test accuracy:{:.5f}, test macro-f1: {:.5f}, model saved at".format( test_acc, test_f1),best_model_path)
                

                #print("{}-GroundTruth Confusion metrics".format(model_type))
                #print(metrics.confusion_matrix(y_test, predicted_y))


#print("svm-tree Confusion metrics".format())
#print(metrics.confusion_matrix(model_preds['svm'], model_preds['tree']))

#print("binarized predictions")
#print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False]))
#print("binarized predictions -- normalized over true labels")
#print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='true'))
#print("binarized predictions -- normalized over pred  labels")
#print(metrics.confusion_matrix(binary_preds['svm'], binary_preds['tree'], labels=[True, False] , normalize='pred'))
        
# print(pd.DataFrame(results).groupby('model_type').describe().T)
