import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import operator
import os
import mediapipe as mp


dir_model= Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
model=tf.keras.models.load_model(dir_model/'model_landmarks_22_03_25.keras', custom_objects={'softmax_v2': tf.nn.softmax})

dir_dataset = Path(r'D:\ASL_Dataset\Train')

# Erstellen des Zielordners, falls er noch nicht existiert
dir_dataset.mkdir(parents=True, exist_ok=True)

#dictionarys erstellen
landmarks_nicht_erkannt={}
true_positives_1={}
true_positives_3={}
false_negatives={}
false_positives={"A":0,"B":0,"C":0,"D":0,"E":0,"F":0,"G":0,"H":0,"I":0,"K":0,"L":0,"M":0,"N":0,"O":0,"P":0,"Q":0,"R":0,"S":0,"T":0,"U":0,"V":0,"W":0,"X":0,"Y":0}
true_negatives={"A":0,"B":0,"C":0,"D":0,"E":0,"F":0,"G":0,"H":0,"I":0,"K":0,"L":0,"M":0,"N":0,"O":0,"P":0,"Q":0,"R":0,"S":0,"T":0,"U":0,"V":0,"W":0,"X":0,"Y":0}

#mediapipe hands model initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

for curr_dir, subdirs, files in os.walk(dir_dataset):
    # nur die richtigen Unterverzeichnisse sind nicht leer
    if len(files) == 0:
        continue

    buchstabe=curr_dir[-1]
    true_positive_1=0
    true_positive_3=0
    false_negative=0
    l_nicht_erkannt=0
    count=0
    for file in files:
        if count>=100:
            continue 

        filepath = curr_dir + os.sep + file
        print(filepath)
        image = cv2.imread(filepath)
        image=cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        hand_landmarks_list_org = []
        h, w, c = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_landmarks_list_org.append((cx, cy))

                hand_landmarks_list=[]

                x_min = min(hand_landmarks_list_org, key=lambda x: x[0])[0]
                x_max = max(hand_landmarks_list_org, key=lambda x: x[0])[0]
                y_min = min(hand_landmarks_list_org, key=lambda x: x[1])[1]
                y_max = max(hand_landmarks_list_org, key=lambda x: x[1])[1]

                delta_y=y_max-y_min
                delta_x=x_max-x_min
    
                landmark_data = [((landmark.x*w-x_min)/delta_x) for landmark in hand_landmarks.landmark] , [((landmark.y*h-y_min)/delta_y)for landmark in hand_landmarks.landmark] 
                
                # Convert landmarks to numpy array and append to list
                landmark_data = np.array(landmark_data)
                hand_landmarks_list.append(landmark_data)
                    
                hand_landmarks_list=np.array(hand_landmarks_list)
                hand_landmarks_list=hand_landmarks_list.reshape(2,21)
    
                predictions = model.predict(np.array([ hand_landmarks_list]))
                #print("prediction", predictions)

                letters=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
                to_sort=[]
                for i in range(24):
                    to_sort.append([predictions[0,i],letters[i]])
                
                to_sort.sort(key=operator.itemgetter(0),reverse=True)
                top_three = to_sort[:3]
                print(top_three)

                #wenn eines der top 3 erkannten Ergebnisse gleich dem Ergebniss ist, was rauskommen soll
                if top_three[0][1]==buchstabe or top_three[1][1]==buchstabe or top_three[2][1]==buchstabe:
                    true_positive_3=true_positive_1+1

                #wenn das erkannte Ergebniss gleich dem Ergebniss ist, was rauskommen soll
                if top_three[0][1]==buchstabe:
                    true_positive_1=true_positive_1+1
                    # für jeden Buchstaben (mit zugehöriger Zahl) im dictionary true_negatives
                    for key, value in true_negatives.items():
                        #wenn der Buchstabe im dictionary nicht der erkannte Buchstabe ist, erhöhe die Zahl dieses Buchstaben um 1
                        if key!=buchstabe:
                            true_negatives[key]+=1

                #wenn das erkannte Ergebniss ungleich dem Ergebniss ist, was rauskommen soll
                #if top_three[0][1]!=buchstabe:
                else:
                    false_negative=false_negative+1
                    #false_positives des falschen Ergebnisses um 1 erhöhen
                    false_positives[top_three[0][1]] +=1 
                    for key, value in true_negatives.items():
                        if key!=buchstabe and key!=top_three[0][1]:
                            true_negatives[key]+=1

                
            
            count=count+1

        else: 
            l_nicht_erkannt+=1
            false_negative=false_negative+1
            for key, value in true_negatives.items():
                        if key!=buchstabe:
                            true_negatives[key]+=1
            count=count+1

                
        

    true_positives_1[buchstabe] = [true_positive_1]
    true_positives_3[buchstabe] = [true_positive_3]
    false_negatives[buchstabe] = [false_negative]
    landmarks_nicht_erkannt[buchstabe]=[l_nicht_erkannt]


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

print("landmarks_nicht_erkannt")
print(landmarks_nicht_erkannt)
