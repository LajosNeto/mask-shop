"""
Facial expression detection model h5 file generator
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import sys
import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")
import argparse

import pandas as pd
import numpy as np
import cv2

from models import list_models, build_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def load_dataset():
    print("====> Preparing data")
    data = pd.read_csv('data/FER2013/fer2013.csv')
    X = []
    for index, row in data.iterrows():
        image_pixels = np.asarray(list(row['pixels'].split(' ')), dtype=np.float32)
        image_pixels = image_pixels.reshape((48,48))
        X.append(image_pixels)
    X = np.array(X)
    X = np.expand_dims(X, 3)

    y = to_categorical(data['emotion'].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mean = np.mean(X)
    std = np.std(X)
    X_test = (X_test - mean)/std

    train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True)
    train_gen.fit(X_train)
    train_generator = train_gen.flow(X_train, y_train, batch_size=128)

    print("====> Finished preparing data")
    return(train_generator, X_test, y_test)

def generate_model(model, model_name):
    train_generator, X_test, y_test = load_dataset()
    print("====> Start training model")
    history = model.fit_generator(train_generator, steps_per_epoch=224, epochs=30, validation_data=(X_test, y_test), verbose=3)
    print("====> Fininshed training model")
    print(f"train acc : {history.history['accuracy'][-1]}\nval acc : {history.history['val_accuracy'][-1]}")
    print(f"train loss : {history.history['loss'][-1]}\nval loss : {history.history['val_loss'][-1]}")
    print("====> Saving model")
    model.save(f"model_output/{model_name}.h5")

if __name__ == '__main__':
    model_generator_argparse = argparse.ArgumentParser()
    model_generator_argparse.add_argument('--model', type=str, required=True)
    args = model_generator_argparse.parse_args()
    model_name = args.model

    model = build_model(model_name)
    if(model):
        print(f"====> Start {model_name} model generation")
        generate_model(model, model_name)
    else:
        print(f"Invalid model, please, provide one of the following options : {', '.join(list_models())}")