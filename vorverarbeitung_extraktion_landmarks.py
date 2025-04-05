

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pickle
import os

dir2 = Path(r'D:\datensatz neu\asl_alphabet_train\asl_alphabet_train')
dir4 = Path(r'D:\pickle\mediapipe\mediapipe_22_3_25')

images_train = []
y_train = []
hand_landmarks_list = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

for subdir, dirs, files in os.walk(dir2):
    for file in files:

        print(file)
       
        image_path = Path(subdir) / file  

        image = cv2.imread(str(image_path)) 
       
        if image is None:
            print(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        hand_landmarks_list_org = []
        h, w, c = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_landmarks_list_org.append((cx, cy))

                if len(hand_landmarks_list_org) == 21:

                    x_min = min(hand_landmarks_list_org, key=lambda x: x[0])[0]
                    x_max = max(hand_landmarks_list_org, key=lambda x: x[0])[0]
                    y_min = min(hand_landmarks_list_org, key=lambda x: x[1])[1]
                    y_max = max(hand_landmarks_list_org, key=lambda x: x[1])[1]

                    delta_y=y_max-y_min
                    delta_x=x_max-x_min
        
                    landmark_data = [((landmark.x*w-x_min)/delta_x) for landmark in hand_landmarks.landmark] , [((landmark.y*h-y_min)/delta_y)for landmark in hand_landmarks.landmark] 
                    
                    landmark_data = np.array(landmark_data)
                    hand_landmarks_list.append(landmark_data)
                       
                    
                    buchstabe = file[0]
                    
                    if ord(buchstabe) < 74:  
                        zahl_von_buchstabe = ord(buchstabe) - 65
                    else:  
                        zahl_von_buchstabe = ord(buchstabe) - 66
                    
                    y_train.append(zahl_von_buchstabe)

        else:
            continue

x_train = np.stack(hand_landmarks_list)

with open(dir4 / 'x_train.pickle', 'wb') as f:
    pickle.dump(x_train, f)

with open(dir4 / 'y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

print("Processing complete. Data saved successfully.")