from flask import Flask, request
from joblib import load

app = Flask(__name__)
# load the model for making predictions
model= load('/mnt/c/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')

@app.route("/predictor/<image>")
def predictor():
    js=request.get_json()
    image=js['image']
    # convert image to array and make prediction using the model 
    classification= model.predict(np.array(image).reshape(1, -1))
       
    return int(classification)    

@app.route("/predict_post/<image>, methods=["POST"])
def predict_post():
    js=request.get_json()
    image=js['image']
    # convert image to array and make prediction using the model 
    classification= model.predict(np.array(image).reshape(1, -1))
    
    return {"op" : int(classification) + request.json["suffix"]}