from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
import itertools
from sklearn.metrics import accuracy_score
# we will put all utils here



def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model


def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test =  train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
    return X_train, X_test, y_train, y_test, X_dev, y_dev

# Question 2:
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )


    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

def split_train_dev_test(X, y, test_size, dev_size):
    ratio_train = 1-(dev_size+test_size)
    ratio_val = dev_size
    ratio_test = test_size

    # Produces test split.
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        X, y, test_size=ratio_test)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    x_train, x_val, y_train, y_val = train_test_split(
        x_remaining, y_remaining, test_size=ratio_val_adjusted)
    
    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)
      
    return x_train, x_val,x_test, y_train,y_val,y_test

# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model

def predict_and_eval_accuracy(model, X, y):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X)
    return  accuracy_score(y, predicted)  

def best_hyperparams(X_train, y_train,X_val, y_val, hparams_combi, model_type="svm"):
    best_accy_sofar=-1
    for hparams in hparams_combi:
        curr_model=train_model(X_train, y_train, {'gamma':hparams[0],'C':hparams[1] }, model_type="svm")
        curr_accy=predict_and_eval_accuracy(curr_model, X_val, y_val)
        if curr_accy>best_accy_sofar:
            best_accy_sofar=curr_accy
            best_model=curr_model
            optimal_gamma=hparams[0]
            optimal_C=hparams[1]
            
            #print(best_accy_sofar)
    return [optimal_gamma,optimal_C],best_model,best_accy_sofar

# fn to output list of all possible combinations of hparams
def hparams_combination(*args): # indicates multi input possible
    # create list of combination of all possible hparams
    hparams_combi= list(itertools.product( *args))
    return hparams_combi
