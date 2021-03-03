from flask import Flask,render_template,request,redirect
from flask.helpers import get_debug_flag
import os,uuid,json,re,time



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

audioPath = 'files'

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
    now = time.time()
    for f in os.listdir(audioPath):
        fp = os.path.join(audioPath,f)
        if os.stat(fp).st_mtime < now - 1 * 60 *60: #older than 1 hour
            if os.path.isfile(fp):
                os.remove(fp)
    return render_template('record.html')

@app.route("/rec",methods=['GET','POST'])
def rec():
    print("Writing sound data...")
    uid = str(uuid.uuid4()) 
    f = open(os.path.join(audioPath,uid+ '.wav'), 'wb')
    f.write(request.data)
    f.close()
    print(f"Sound data written to '{uid}'.")
    return json.dumps({'uuid':uid})

@app.route("/result/<uid>",methods=['GET','POST'])
def result(uid):
    print("result:"+uid)
    regex = re.compile('^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z', re.I)
    if regex.match(uid):
    
        if os.path.isfile(os.path.join(audioPath,uid+ '.wav')):
            # symbol=request.form['symbol']
            
            result=model(array (mel_spec(os.path.join(audioPath,uid+ '.wav'))))
            #os.remove(os.path.join(audioPath,uid+ '.wav'))

            return render_template ('result.html',result=result)
        
            
    return redirect("/")
@app.route("/result",methods=['GET','POST'])
def result_blank():
    return redirect("/")


if __name__ == "__main__":
    app.run(host= '0.0.0.0')
    #ui.run()
