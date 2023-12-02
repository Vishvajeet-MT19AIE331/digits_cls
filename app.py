from flask import Flask, request
import numpy as np
from joblib import load

app = Flask(__name__)
# load the model for making predictions
#model= load('C:/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')
model= load('/mnt/c/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')
#model= load('./models/svm_gamma_0.001_C_1.joblib')

@app.route("/<modeltype>")
def load_model(modeltype):
    if modeltype == 'svm':
        model= load('./models/svm_gamma_0.001_C_1.joblib')
    elif modeltype == 'tree':
        model= load('./models/tree_max_depth_10.joblib')
    else:
        model= load('./models/MT19AIE331_lr_lbfgs_.joblib')


@app.route("/", methods=["POST"])
def predict_post(modeltype):
    
    js=request.get_json()
    image=js['image']
    # convert image to array and make prediction using the model 
    classification= model.predict(np.array(image).reshape(1, -1))
    
    return {"The Model predicts that the image belongs to Class" : int(classification) }


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)