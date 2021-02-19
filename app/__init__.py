from flask import Flask,render_template,request
from flask.helpers import get_debug_flag


# from scipy.io.wavfile import write


import librosa

import librosa.display

import numpy as np
import cv2
import keras
from keras import models
from keras import backend as K
# from IPython.display import display
import matplotlib.pyplot as plt

# from flask_jsonpify import jsonpify

app = Flask(__name__)

# Feed it the flask app instance 
# ui = FlaskUI(app)



def mel_spec(path):
    y, sr = librosa.load(path)
    librosa.feature.melspectrogram(y=y, sr=sr)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        fmax=8000)
    plt.figure(figsize=(10,8))
    librosa.display.specshow(librosa.power_to_db(S,
                                                 ref=np.max),
                             fmax=8000,
                              )
    plt.savefig("test/test.png")
    return("test/test.png")
def array (impath):
    img=cv2.imread(impath)
    img=cv2.resize(img,(200,200))
    return img
def model(arr):
    reconstructed_model = keras.models.load_model("amodel")
    result=reconstructed_model.predict_classes(arr.reshape(-1, 200, 200,3),verbose=2)
    if result == 1:
        return("genuine")
    else:
        return("fraud")

@app.route("/")
def home():
    return render_template('record.html')

@app.route("/rec",methods=['GET','POST'])
def rec():
    f = open('./file.wav', 'wb')
    f.write(request.data)
    f.close()
    return "Binary message written!"

@app.route("/result",methods=['GET','POST'])
def result():
    if request.method == 'POST':
        # symbol=request.form['symbol']
        
        result=model(array (mel_spec("file.wav")))
        

        return render_template ('result.html',result=result)
    else:
        return render_template('index_old.html')




if __name__ == "__main__":
    app.run()
    #ui.run()