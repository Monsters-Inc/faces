import os
import dlib
import cv2
import numpy as np
from preprocessing_helpers import median_filtering_single, canny_edges_single, clahe_single, he_single, format_folder_name

#
# Get cropped faces from image
#
def get_faces(image, filename, wrong_path):

    model=dlib.cnn_face_detection_model_v1('../../mmod_human_face_detector.dat')
    faces=model(image)
    cropped=[]

    if len(faces) == 0:
        cv2.imwrite(wrong_path+filename, image)
    else:

        #for face in faces:
        face = faces[0]
        x=face.rect.left()
        y=face.rect.top()
        w=face.rect.right()
        h=face.rect.bottom()
        x=max(0, x)
        y=max(0, y)
        w=max(0, w)
        h=max(0, h)

        crop=image[y:h, x:w]
        cropped.append(crop)

    return cropped

#
# Preprocess an image with 0 or more preprocessing methods
#
def preprocess(input_data, end_path, wrong_path, img_shape, preprocessing):
    faces = []
  
    input_data = format_folder_name(input_data)
    folder = sorted(os.listdir(input_data))
    filenames = folder

    images = []
    for file in folder:
        img = cv2.imread(input_data + file, 0)
        images.append(img)

    faces = []

    for i, image in enumerate(images):
        cropped_faces = get_faces(image, filenames[i], wrong_path)

        if len(cropped_faces) != 0:
            faces = faces + cropped_faces

    for i in range(len(faces)):
        faces[i] = cv2.resize(faces[i], (img_shape[0], img_shape[1]))

        if len(preprocessing) > 0:
            if 'he' in preprocessing:
                faces[i] = he_single(faces[i])

            if 'clahe' in preprocessing:
                faces[i] = clahe_single(faces[i])

            if 'canny' in preprocessing:
                faces[i] = canny_edges_single(faces[i])
            
            if 'median' in preprocessing:
                faces[i] = median_filtering_single(faces[i])

    #faces = np.array(faces)

    # Save the faces in end_path
    #print('Length faces: ', len(faces))
    for i, face in enumerate(faces):
        cv2.imwrite(end_path+filenames[i], face)

