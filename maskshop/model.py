"""
Facial expression detection model
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# import logging
# logging.getLogger("tensorflow").setLevel(logging.WARNING)

import pandas as pd
import numpy as np

import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, AvgPool2D, Dropout, Flatten, Activation, ReLU, BatchNormalization
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


EMOTION_LABELS = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}


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

    train_gen = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   horizontal_flip=True)
    train_gen.fit(X_train)
    train_generator = train_gen.flow(X_train, y_train, batch_size=128)
    print("====> Finished preparing data")
    return(train_generator, X_test, y_test)

def build_model():
    model = Sequential(name='Custom 2')
    
    model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(512, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(512, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(256, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(7, activation='softmax', name='Output'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model

def generate_model():
    train_generator, X_test, y_test = load_dataset()
    model = build_model()
    print("====> Start training model")
    history = model.fit_generator(train_generator, steps_per_epoch=224, epochs=30, validation_data=(X_test, y_test), verbose=3)
    print("====> Fininshed training model")
    print(f"train acc : {history.history['accuracy'][-1]}\nval acc : {history.history['val_accuracy'][-1]}")
    print(f"train loss : {history.history['loss'][-1]}\nval loss : {history.history['val_loss'][-1]}")
    print("====> Saving model")
    model.save('model_output/face_detection.h5')

if __name__ == '__main__':
    generate_model()