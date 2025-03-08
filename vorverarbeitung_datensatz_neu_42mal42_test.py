import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt

dir_dataset = Path(r'D:\datensatz neu\asl_alphabet_test\asl_alphabet_test')
dir_edited_dataset = Path(r'D:\datensatz neu 42mal42\test')

# Erstellen des Zielordners, falls er noch nicht existiert
dir_edited_dataset.mkdir(parents=True, exist_ok=True)

for subdir, dirs, files in os.walk(dir_dataset):
    for file in files:
        filepath = subdir + os.sep + file
        print(filepath)
        bild = cv2.imread(filepath)

        # Anzeigen des Originalbildes
        #plt.imshow(cv2.cvtColor(bild, cv2.COLOR_BGR2RGB))  # OpenCV liest in BGR, aber matplotlib erwartet RGB
        #plt.show()

        # Umwandlung in Graustufen
        gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

        # Bild auf 28x28 Größe ändern
        resized_image = cv2.resize(gray, (42, 42))

        # Normalisierung des Bildes (0 bis 255)
        resized_image = cv2.normalize(resized_image, resized_image, 0, 255, cv2.NORM_MINMAX)

        # Bild speichern (optional)
        cv2.imwrite(str(dir_edited_dataset / file), resized_image)