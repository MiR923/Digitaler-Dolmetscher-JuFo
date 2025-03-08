import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import operator
import os



dir_model= Path(r'C:\Users\BEN UTZER\Documents\Milena\JuFo\Fertiges Netz')
model=tf.keras.models.load_model(dir_model/'model_42mal42_6000.keras', custom_objects={'softmax_v2': tf.nn.softmax})

dir_dataset = Path(r'C:\Users\BEN UTZER\Documents\Milena\JuFo\DatensatzG\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\testedit')

# Erstellen des Zielordners, falls er noch nicht existiert
dir_dataset.mkdir(parents=True, exist_ok=True)

true_positives_1={}
true_positives_3={}
false_negatives={}
false_positives={"A":0,"B":0,"C":0,"D":0,"E":0,"F":0,"G":0,"H":0,"I":0,"K":0,"L":0,"M":0,"N":0,"O":0,"P":0,"Q":0,"R":0,"S":0,"T":0,"U":0,"V":0,"W":0,"X":0,"Y":0}
true_negatives={"A":0,"B":0,"C":0,"D":0,"E":0,"F":0,"G":0,"H":0,"I":0,"K":0,"L":0,"M":0,"N":0,"O":0,"P":0,"Q":0,"R":0,"S":0,"T":0,"U":0,"V":0,"W":0,"X":0,"Y":0}
for curr_dir, subdirs, files in os.walk(dir_dataset):
    # nur die richtigen Unterverzeichnisse sind nicht leer
    if len(files) == 0:
        continue

    buchstabe=curr_dir[-1]
    true_positive_1=0
    true_positive_3=0
    false_negative=0
    count=0
    for file in files:
        if count>=100:
            continue 

        filepath = curr_dir + os.sep + file
        print(filepath)
        bild = cv2.imread(filepath)
        bild=cv2.flip(bild, 1)


        # Umwandlung in Graustufen
        gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

        # Bild auf 28x28 Größe ändern
        resized_image = cv2.resize(gray, (42,42))

        # Normalisierung des Bildes (0 bis 255)
        resized_image = cv2.normalize(resized_image, resized_image, 0, 255, cv2.NORM_MINMAX)
        resized_image = resized_image / 255.0

        predictions = model.predict(np.array([resized_image]))
        print("prediction", predictions)

        letters=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
        to_sort=[]
        for i in range(24):
            to_sort.append([predictions[0,i],letters[i]])
        
        to_sort.sort(key=operator.itemgetter(0),reverse=True)
        top_three = to_sort[:3]
        print(top_three)

        #wenn das erkannte Ergebniss gleich dem Ergebniss ist, was rauskommen soll
        if top_three[0][1]==buchstabe:
            true_positive_1=true_positive_1+1
            # für jeden Buchstaben (mit zugehöriger Zahl) im dictionary true_negatives
            for key, value in true_negatives.items():
                #wenn der Buchstabe im dictionary nicht der erkannte Buchstabe ist, erhöhe die Zahl dieses Buchstaben um 1
                if key!=buchstabe:
                    true_negatives[key]+=1

        #wenn das erkannte Ergebniss ungleich dem Ergebniss ist, was rauskommen soll
        if top_three[0][1]!=buchstabe:
            false_negative=false_negative+1
            #false_positives des falschen Ergebnisses um 1 erhöhen
            false_positives[top_three[0][1]] +=1 
            for key, value in true_negatives.items():
                if key!=buchstabe and key!=top_three[0][1]:
                    true_negatives[key]+=1

         #wenn eines der top 3 erkannten Ergebnisse gleich dem Ergebniss ist, was rauskommen soll
        if top_three[0][1]==buchstabe or top_three[1][1]==buchstabe or top_three[2][1]==buchstabe:
            true_positive_3=true_positive_1+1

        count=count+1

    true_positives_1[buchstabe] = [true_positive_1]
    true_positives_3[buchstabe] = [true_positive_3]
    false_negatives[buchstabe] = [false_negative]


print("true_positives_1")
print(true_positives_1)

print("true_positives_3")
print(true_positives_3)

print("false_negatives")
print(false_negatives)

print("true_negatives:")
print(true_negatives)

print("false_positives")
print(false_positives)