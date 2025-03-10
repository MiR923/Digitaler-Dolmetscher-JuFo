import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import tensorflow as tf
import operator

dir_model = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
model = tf.keras.models.load_model(dir_model / 'model_neuer_datensatz_42mal42.keras', custom_objects={'softmax_v2': tf.nn.softmax})

# Mediapipe Vorbereitung
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    a = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
               
                    # Region of Interest
                hand_landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_landmarks_list.append((cx, cy))

                if len(hand_landmarks_list) > 0:
                    x_min = min(hand_landmarks_list, key=lambda x: x[0])[0]
                    x_max = max(hand_landmarks_list, key=lambda x: x[0])[0]
                    y_min = min(hand_landmarks_list, key=lambda x: x[1])[1]
                    y_max = max(hand_landmarks_list, key=lambda x: x[1])[1]

                    delta_y=y_max-y_min
                    delta_x=x_max-x_min

                    delta = int(np.ceil(max(delta_x, delta_y)/2))
                    x_center = int(np.ceil((x_max+x_min)/2))
                    y_center = int(np.ceil((y_max+y_min)/2))

                    roi = image[((y_center-delta)-25):((y_center+delta)+25), (x_center-delta-25):(x_center+delta+25)]
                    cv2.rectangle(image, ((x_center-delta-25), (y_center-delta-25)), ((x_center+delta+25), (y_center+delta+25)), (0, 255, 0), 2)
                
                    if roi.size > 0: #Check if roi is not empty
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        resized_image = cv2.resize(gray, (42, 42))
                        resized_image = cv2.normalize(resized_image, resized_image, 0, 255, cv2.NORM_MINMAX)
                        resized_image = resized_image / 255.0

                        cv2.imshow('resized_image',resized_image)
                        #print(hand_landmarks_list)
                        
                        predictions = model.predict(np.array([resized_image]))
                        
                        letters=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
                        to_sort=[]
                        for i in range(24):
                            to_sort.append([predictions[0,i],letters[i]])
                            
                        to_sort.sort(key=operator.itemgetter(0),reverse=True)
                        top_one=to_sort[0]
                        top_three = to_sort[:3]

                        if np.max(predictions) < 0.3: 
                            predicted_letter = "Nicht erkannt"
                            print("Nicht erkannt")
                        else:
                            predicted_letter = top_one[1]
                            #print(predictions)
                            print(top_one)
                            print (top_three)

                            text_size = cv2.getTextSize(predicted_letter, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                            text_x, text_y = 50, 50
                            box_x, box_y = text_x - 5, text_y + 5
                            box_w, box_h = text_size[0] + 10, -text_size[1] - 10
                            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
                            cv2.putText(image, predicted_letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()