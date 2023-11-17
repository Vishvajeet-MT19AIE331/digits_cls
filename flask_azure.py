from flask import Flask, request
import numpy as np
from joblib import dump, load

app = Flask(__name__)


#model= load('C:/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')
model= load('/mnt/c/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')


@app.route("/pred", methods=['POST'])

def predictor():
    js=request.get_json()
    image=js['image']
    # convert image to array and make prediction using the model for both images
    classification= model.predict(np.array(image).reshape(1, -1))
       
    return int(classification)
