import os
fldr="test_pictures"
#import sys
#fldr="pictures"
#files=os.listdir(fldr)
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
#face_cascade_profile_right = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

##Kör bara en bild just nu. Gör en for-loop för att lägga in flera bilder i lista
# img_name = sys.argv[1]
# img = cv2.imread(fldr+ '/' + img_name + '.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_crop = []
for filename in os.listdir(fldr):
    img = cv2.imread(fldr+'/' + filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        faces = face_cascade_profile.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        flipped = cv2.flip(gray_img, 1)
        faces = face_cascade_profile.detectMultiScale(flipped, 1.3, 5)
    if len(faces) == 0:
        faces = alt.detectMultiScale(gray_img, 1.3, 5)
    #if len(faces) == 0:
       # faces = alt2.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        faces = alt_tree.detectMultiScale(gray_img, 1.3, 5)

    ## We want the face with the largest area
    sqr_area = 0
    index = 0
    biggest_area_index = 0 ## index of face with biggest area
    for (x,y,w,h) in faces:
        current_area = w*h
        if sqr_area < current_area:
            biggest_area_index = index
            sqr_area = current_area
        index+=1

    if (len(faces) != 0):      
        x = faces[biggest_area_index][0]
        y = faces[biggest_area_index][1]
        w = faces[biggest_area_index][2]
        h = faces[biggest_area_index][3]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        face_crop.append(img[y:y+h, x:x+w])
        #print(img[y:y+h, x:x+w])
        #print("PING")
    
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    else:
        print(filename)



##cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

##Lägg i en for-loop för flera bilder
print("TOTAL LENGTH")
print(len(face_crop))
for i in face_crop:
    cv2.imshow('face',i)
    cv2.waitKey(0)


