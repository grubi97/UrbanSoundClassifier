import librosa
import librosa.display
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import glob
import sys
import re
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image



app = Flask(__name__)
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))
sound=""
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
     return render_template('index.html')

@app.route('/predict',methods=['Post'])
def predict():
     categories=['air conditioner','car horn','child playing','dog bark','drilling','engine idling','gun shot','jackhammer','siren','music']
     Fs = 41000
     d = 5
     print('Speak')
     record = sd.rec(int(d*Fs), Fs, 1, blocking=True) 
     print('Stop')
     sf.write('rec.wav',record,Fs)
     audio, sample_rate = librosa.load('rec.wav', res_type='kaiser_fast')
     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
     mfccs_processed = np.mean(mfccs.T, axis=0)
     X=np.array([mfccs_processed])
     prediction=model.predict_classes(X)
   
     return render_template('index.html',prediction_text=categories[prediction[0]])


app.config['SOUND_PATH']='sounds/'

@app.route('/upload',methods=['GET','POST'])
def upload():

          if request.method=="POST":
               if request.files:
                    sound=request.files["sound"]
                    sound.save(os.path.join(app.config["SOUND_PATH"],sound.filename))
                    categories=['air conditioner','car horn','child playing','dog bark','drilling','engine idling','gun shot','jackhammer','siren','music']
                    audio, sample_rate = librosa.load(os.path.join(app.config["SOUND_PATH"],sound.filename), res_type='kaiser_fast')
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    mfccs_processed = np.mean(mfccs.T, axis=0)
                    X=np.array([mfccs_processed])
                    prediction=model.predict_classes(X)

                    return render_template('index.html',prediction_text=categories[prediction[0]])




        
          
          

if __name__=='__main__':
    
     app.run(threaded=False)