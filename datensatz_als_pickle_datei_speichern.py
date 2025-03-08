import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
import pickle

dir1 = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\fertige neuronale netze')
dir2 = Path(r'D:\datensatz neu\edited dataset')
dir3 = Path(r'D:\datensatz neu\edited_dataset_test')
dir4 = Path(r'C:\Users\Lisa\OneDriveLisa\OneDrive\Dokumente\Schule Lisa\Seminararbeit\pickle')

images_train=[]
y_train=[]

for subdir, dirs, files in os.walk(dir2):
    for file in files:
        #dateiname ausgeben
        print(file)
        #Bild aud Datei laden
        image = cv2.imread(dir2/file)
        #RGB darstellung der Graustufen in wirkliche Graustufendarstellung (nur ein Helligkeitswert) umwandeln
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Graustufen Werte zwischen 0 und 1 bekommen
        image = image / 255
        #Bild in numpy array umwandeln und Dimension hinzufügen (weil im neuronalen Netz gefordert)
        np_img = np.array( image).reshape(28,28,1)
        #array des jeweiligen Bildes in array "images" hinzufügen
        images_train.append( np_img)
        #erstes Zeichen des Strings "file" entspricht dem Buchstaben 
        buchstabe=file[0]

        #ord() wandelt buchstabe in entsprechende Zahl des ASCII codes um
        #J (was ja nicht drin ist) entspricht 74
        if ord(buchstabe)<74:
            #dadurch wird A zu 0, B zu 1 usw.
            zahl_von_buchstabe=ord(buchstabe)-65
        else:
            #dadurch wird K zu 9, L zu 10 usw.
            zahl_von_buchstabe=ord(buchstabe)-66

        #Buchstabe des jeweiligen Bildes in array "" hinzufügen
        y_train.append(zahl_von_buchstabe)       

x_train=np.stack(images_train)

with open(dir4/'x_train.pickle','wb') as f:
    pickle.dump(x_train, f)

with open(dir4/'y_train','wb') as f:
    pickle.dump(y_train, f)