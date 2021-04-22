import cv2
import numpy as np
import pandas as pd
import os

start_path = 'processed_age_and_gender_pictures/'
end_path = 'cropped_faces/'
wrong_path = 'wrong_faces/'


# lägger till en kommentar här för att fixa skit med git -.-
# for f in os.listdir(end_path):
#     os.remove(f)
##Import cascade files for the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

wrongs = 0
##Try the different haarcascade files to see which works for the picture
for filename in os.listdir(start_path):
    img = cv2.imread(start_path+'/' + filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## cv2.COLOR_BGR2RGB ? 
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    
    if len(faces) == 0:
        faces = face_cascade_profile.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        flipped = cv2.flip(gray_img, 1)
        faces = face_cascade_profile.detectMultiScale(flipped, 1.3, 5)
    if len(faces) == 0:
        faces = alt.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        faces = alt2.detectMultiScale(gray_img, 1.3, 5)
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
        img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),2)

        img_cropped = img[y:y+h, x:x+w]
        img_resized = cv2.resize(img_cropped,(48,48))
        
        print(end_path+filename)
<<<<<<< HEAD
        #print(img_resized)
=======
>>>>>>> 9ef4d0c05e70347daa0eb640fb22f8a296e5becf
        cv2.imwrite(end_path+filename, img_resized)

    else:
        cv2.imwrite(wrong_path+filename, img)
        wrongs+=1
        print('ERROR')
