from flask import Flask, request

app = Flask(__name__)

# load from volume
model= load('/mnt/c/Users/vishv/OneDrive/Documents/GitHub/digits_cls/models/svm_gamma_0.001_C_1.joblib')
# take user input as two hand written digit images
@app.route("/image_compar/<image1>/<image2>", methods=['POST'])
def image_comparer(image1,image2):
    # convert image to array and make prediction using the model for both images
    classification_1= model.predict(np.array(image1))
    classification_2= model.predict(np.array(image2))
    # compare the two predictions and return True if both same
    if classification_1 == classification_2:
        return True
    else:
        return False