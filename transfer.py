#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:38:09 2017
CIFAR : Transfer Learning
@author: bbhask002c
"""
import pickle
import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model

epochs = 50 
batch_size = 256


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val
    
    
def main(_):
    # load bottleneck data
    training_file = 'resnet-100/resnet_cifar10_100_bottleneck_features_train.p'
    validation_file = 'resnet-100/resnet_cifar10_bottleneck_features_validation.p'
    X_train, y_train, X_val, y_val = load_bottleneck_data(training_file, validation_file)

    classes = len(y_train)
    input_shape = X_train.shape[1:]
    inp = Input(shape = input_shape)
    x = Flatten()(inp)
    x = Dense(classes, activation = 'softmax')(x)
    model = Model(inp,x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), shuffle=True)
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()    
    
