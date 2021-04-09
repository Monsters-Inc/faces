import os
import sys
fldr="pictures"
#files=os.listdir(fldr)

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

##Kör bara en bild just nu. Gör en for-loop för att lägga in flera bilder i lista
img_name = sys.argv[1]
img = cv2.imread(fldr+ '/' + img_name + '.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_crop = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    face_crop.append(img[y:y+h, x:x+w])
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

##cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

##Lägg i en for-loop för flera bilder
cv2.imshow('face',face_crop[0])
cv2.waitKey(0)
