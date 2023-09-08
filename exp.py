"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""



# Import datasets, classifiers and performance metrics
import numpy as np
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score
import itertools
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split,split_train_dev_test,train_model,predict_and_eval_accuracy,best_hyperparams,hparams_combination

# 1. Get the dataset
X, y = read_digits()

# lists of hparams
gamma=[.001,.01,.1,1,10,100]
C_value=[.1,1,2,5,10]

hparams_combi=hparams_combination(gamma,C_value)

# Vary the dataset sizes
test_size_list=[.1,.2,.3]
val_size_list=[.1,.2,.3]

data_size_combi=hparams_combination(test_size_list,val_size_list)

for data_size in data_size_combi:
    X_train,X_val,X_test,y_train,y_val,y_test=split_train_dev_test(X,y,test_size=data_size[0],dev_size=data_size[1])
    best_hparams, best_model, best_accuracy=best_hyperparams(X_train, y_train,X_val, y_val, hparams_combi, model_type="svm")
    train_acc=predict_and_eval_accuracy(best_model, X_train, y_train)
    test_acc=predict_and_eval_accuracy(best_model, X_test, y_test)
    
    print(f"test_size={np.round(data_size[0],3)} dev_size={np.round(data_size[1],3)} train_size={np.round(1-(data_size[0]+data_size[1]),3)} train_acc={np.round(train_acc,3)} dev_acc={np.round(best_accuracy,3)} test_acc={np.round(test_acc,3)}")
    print(f"best hyper parameters={best_hparams}") 
    print("_____________________________________________________________________________________________")