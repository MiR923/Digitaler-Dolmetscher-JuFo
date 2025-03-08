
import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from pathlib import Path
import cv2
import pickle

#siehe: https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy

dir1 = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
dir4 = Path(r'D:\pickle\42mal42')

with open(dir4/'x_train.pickle','rb') as f:
    x_train = pickle.load(f)

with open(dir4/'y_train','rb') as f:
    y_train = pickle.load(f)

with open(dir4/'x_test.pickle','rb') as f:
    x_test = pickle.load(f)

with open(dir4/'y_test','rb') as f:
    y_test = pickle.load(f)

#Vorverarbeitung Bilder
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

#Model trainieren
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (42,42,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(tf.keras.layers.Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(tf.keras.layers.Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units = 512 , activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

history=model.fit(x_train,y_train,shuffle=True, batch_size=128,epochs=20,validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

model.save(dir1/'model_neuer_datensatz_42mal42.keras')