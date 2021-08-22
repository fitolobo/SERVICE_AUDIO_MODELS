### Exectue the docker container with make run, then execute the app with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Needed pyhton libraries
### La idea seria utilizar una imagen liviana de torch (no necesito todo torch)
import torch
import torch.nn as nn
### Podrian estar en dos contenedores diferentes
### donde la salida de uno alimenta a otro!
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.python.keras.utils import generic_utils
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model

### Probablemente estas librerias de audio pueden evitarse
### Tomando desde backend el audio con las especificaciones requeridas
### en relacion a la calidad de audio. El modelo tambien puede llegar a cambiar,
### respecto a la calidad de audio que backend soporta para subir.
import librosa
import IPython
from IPython.display import Audio
import warnings; warnings.filterwarnings('ignore')

### Numpy tambien es una alternativa que puede ser evitada aparentemente
### pues podriamos tener el audio preprocesado evitando el uso de arreglos
import numpy as np
import os
import glob
# Model and Emotion Dictionary
import utils
from utils import dividiendo_audio, mfcc_input_audio

# Fastapi
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from scipy import stats

################################################################################
# Title
app = FastAPI(title="Audio Emotion Recognition Service")
################################################################################

################################################################################
# RAVDESS+handcraft tagged examples + Parallel is all you need modified to binary classification
binary_emotions_dict ={
    '0':'no-neutral',
    '1':'neutral',
}

mc_emotions_dict = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}
################################################################################

################################################################################
#### First Model: Neutral - No-neutral ####
class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__()

        ################ TRANSFORMER BLOCK #############################
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        self.conv2Dblock2 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ################# FINAL LINEAR BLOCK ####################
        self.fc1_linear = nn.Linear(512*2+40,num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self,x):

        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        ########## 4-encoder-layer Transformer block w/ 40-->512-->40 feedfwd network ##############
        x_maxpool = self.transformer_maxpool(x)
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        x = x_maxpool_reduced.permute(2,0,1)
        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim=0)

        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)

        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)

        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax
################################################################################
################################################################################

def load_model_mc():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    model_mc = Sequential()
    model_mc.add(Conv1D(128, 5,padding='same',
                     input_shape=(40,1)))
    model_mc.add(Activation('relu'))
    model_mc.add(Dropout(0.1))
    model_mc.add(MaxPooling1D(pool_size=(8)))
    model_mc.add(Conv1D(128, 5,padding='same',))
    model_mc.add(Activation('relu'))
    model_mc.add(Dropout(0.1))
    model_mc.add(Flatten())
    model_mc.add(Dense(8))
    model_mc.add(Activation('softmax'))

    return model_mc

################################################################################
################################################################################
### Modelo Binario ###
################################################################################
device = torch.device("cpu")
model = parallel_all_you_want(len(binary_emotions_dict))
model.load_state_dict(torch.load('models/best_model_binary.pth', map_location=device))
################################################################################
### Modelo Multiclase ###
tf.keras.backend.set_learning_phase(0)
model_mc = tf.keras.models.load_model("models/Emotion_Voice_Detection_Model.h5")
print(model_mc.summary())
################################################################################
# Prediccion del modelo (un audio por vez)
def model_prediction(mfcc_feature_per_audio,emotions_dict):
    model.eval()
    batch_prediction = []
    total_emotions_predicted = []

    #print(len(mfcc_feature_per_audio))

    for i in range(len(mfcc_feature_per_audio)):
        X_valid = np.expand_dims(np.expand_dims(np.array(mfcc_feature_per_audio[i]),1),1)
        X_valid = np.reshape(X_valid,[1,1,40,282])
        X_valid_tensor = torch.tensor(X_valid,device=device).float()
        output_logits, output_softmax = model(X_valid_tensor)
        predictions = torch.argmax(output_softmax,dim=1)
        batch_prediction.append(predictions)
    for item in range(len(batch_prediction)):
        total_emotions_predicted.append(emotions_dict[str(np.array(batch_prediction[item][0]))])

    return total_emotions_predicted

################################################################################
# Substituyendo la no neutralidad
################################################################################
def no_neutral_filter(binary_emotion,input_mean):
    # Modo evaluación
    tf.keras.backend.set_learning_phase(0)

    # Modifico las dimensiones para entrar en el modelo no binario
    input_mean = np.expand_dims(np.asarray(input_mean), axis=2)

    for i in range(len(binary_emotion)):
        if binary_emotion[i] != 'neutral':
            binary_emotion[i] = mc_emotions_dict[str(model_mc.predict_classes(input_mean)[0])]
    return binary_emotion

# API
@app.post("/emotion-recognition")
async def emotion_recognition(file: UploadFile = File(...)):

        sample_rate = 48000
        audio_received,sr = librosa.load(file.file,sr = sample_rate)
        audio_batches = dividiendo_audio(audio_received,sr = sample_rate, audio_input_dim = 3)
        input, input_mean = mfcc_input_audio(audio_batches,sr)

        # Emoción binaria
        binary_emotion = model_prediction(input,binary_emotions_dict)

        # Emocion Multiclase en caso de existir
        final_emotions = no_neutral_filter(binary_emotion,input_mean)


        return {"Emotions detected every 3 sec":final_emotions}
