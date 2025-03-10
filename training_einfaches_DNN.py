
import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from pathlib import Path
import pickle

dir1 = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
dir4 = Path(r'D:\pickle')

with open(dir4/'x_train.pickle','rb') as f:
    x_train = pickle.load(f)
x_train=np.array(np.squeeze(x_train))

with open(dir4/'y_train','rb') as f:
    y_train = np.array(pickle.load(f))

with open(dir4/'x_test','rb') as f:
    x_test = np.array(pickle.load(f))
x_test=np.squeeze(x_test)

with open(dir4/'y_test','rb') as f:
    y_test = np.array(pickle.load(f))

# TF Bilderkennungsmodell
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(24, activation=tf.nn.softmax)
])

# Crossentropy für die 24 Zahlen Klassen
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modellfitting und Evaluation
model.fit(x_train, y_train, epochs=20)
#print(y_test[:10])
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
#print(y_test[:10])
#print(y_test[:10])
  # Save the model in SavedModel format
model.save(dir1/'model_haende_it.keras')
