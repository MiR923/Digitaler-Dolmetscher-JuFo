import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

dir1 = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
dir4 = Path(r'D:\pickle\mediapipe\mediapipe_22_3_25')

with open(dir4/'x_train.pickle','rb') as f:
    x = pickle.load(f)
x=np.array(np.squeeze(x))

with open(dir4/'y_train.pickle','rb') as f:
    y = np.array(pickle.load(f))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.002, random_state=31)
# TF Bilderkennungsmodell
print(x_train.shape)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(21,2)),
  #tf.keras.layers.Flatten()
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
model.save(dir1/'model_landmarks_26_03_25.keras')
