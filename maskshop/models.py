"""
Facial expression detection models
"""

# Author:
# Lajos Neto <lajosneto@gmail.com>


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, AvgPool2D, Dropout, Flatten, Activation, ReLU, BatchNormalization


MODELS = ['lenet_5', 'custom_1', 'custom_2', 'custom_3']

def list_models(): 
    return MODELS

def build_model(name):
    return {
        'lenet_5': __build_lenet5,
        'custom_1': __build_custom1,
        'custom_2': __build_custom2,
        'custom_3': __build_custom3
    }.get(name, __default)()

def __default():
    return None

def __build_lenet5():
    model = Sequential(name='LeNet-5')

    model.add(Conv2D(6, (5,5), strides=(1,1), input_shape=(48,48,1), name='C1', activation='relu'))    
    model.add(AvgPool2D(pool_size=(2,2), name='S2'))
    
    model.add(Conv2D(16, (5,5), strides=(1,1), name='C3', activation='relu'))
    model.add(AvgPool2D(pool_size=(2,2), name='S4'))
    
    model.add(Conv2D(120, (4,4), strides=(1,1), name='C5', activation='relu'))
    model.add(Flatten(name='C5-Flat'))
    model.add(Dense(120, name='C5-Dense', activation='relu'))

    model.add(Dense(84, name='F6'))
    model.add(Dense(7, activation='softmax', name='Output'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model

def __build_custom1():
    model = Sequential(name='Custom 1')

    model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(128, (3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(128, (3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(128, (3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(7, activation='softmax', name='Output'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model

def __build_custom2():
    model = Sequential(name='custom_2')
    
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

def __build_custom3():
    model = Sequential(name='custom_3')
    
    model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(512, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(512, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(256, (3,3), strides=(1,1), input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(7, activation='softmax', name='Output'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model