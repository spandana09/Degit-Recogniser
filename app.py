# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
from PIL import Image
# for matrix math
import numpy as np
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
import cv2
# tell the app where saved model is
sys.path.append(os.path.abspath("./model"))
from load import *
# initalize the flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()
import base64
# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model,
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = Image.open('output.png')
    # make it the right size
    x = x.resize((28, 28))
    x.save('final_image.png')
    # convert to a 4D tensor to feed into the model
    #converting to grayscale
    imgUMat=np.float32(x)
    x=cv2.cvtColor(imgUMat,cv2.COLOR_RGB2GRAY)
    #converting to ndarray to reshape
    x1=np.array(x)
    x1 = x1.reshape((1, 28, 28, 1))
    # in the computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x1)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])
if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=5000)
# optional if you want to run in debugging mode
# app.run(debug=True)