# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd
import pytest

test_size =  .2
dev_size  =  .2
# Run 1
X_train1, X_test1, X_dev1, y_train1, y_test1, y_dev1 = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
# Run 2
X_train2, X_test2, X_dev2, y_train2, y_test2, y_dev2 = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

for i in range(5):
  def test_comparedata():
      assert y_train1[i] == y_train2[i]