import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 



fldr="test_pictures"
df = pd.read_csv('data/dataset.csv', sep=';')

##Import cascade files for the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

##Kör bara en bild just nu. Gör en for-loop för att lägga in flera bilder i lista

##Try 
face_crop = []
genders = []
ages = []

##Try the different haarcascade files to see which
for filename in os.listdir(fldr):
    img = cv2.imread(fldr+'/' + filename)
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
        

        row = df.loc[df['image'] == filename]
        gender = row['gender'].values[0]
        age = row['age'].values[0]

        if gender == 'K':
            genders.append(1)
        elif gender == 'M':
            genders.append(0)
        else:
            print('NO_GENDER_ERROR')

        if not np.isnan(age):
            ages.append(int(age))
        else:
            ages.append(int(40.0))
            print('NO_AGE_ERROR')

        img_cropped = img[y:y+h, x:x+w]
        img_resized = cv2.resize(img_cropped,(48,48))
        face_crop.append(img_resized)

    else:
        print('ERROR')
        print(filename)

print('WOMEN:')
print(genders.count(1))
print('MEN:')
print(genders.count(0))
print("TOTAL LENGTH")
print(len(face_crop))
#for i in face_crop:
#    cv2.imshow('face',i)
#    cv2.waitKey(0)
   #cv2.destroyAllWindows()

face_crop_f = np.array(face_crop)
genders_f = np.array(genders)
ages_f = np.array(ages)


np.save('face_crop.npy',face_crop_f)
np.save('genders.npy',genders_f) 
np.save('ages.npy',ages_f) 

labels=[]
i=0
while i<len(ages):
  label=[]
  label.append([ages[i]])
  label.append([genders[i]])
  labels.append(label)
  i+=1

face_crop_f_2=face_crop_f/255
labels_f = np.array(labels) 

X_train, X_test, Y_train, Y_test= train_test_split(face_crop_f_2, labels_f,test_size=0.25)

Y_train_2=[Y_train[:,1],Y_train[:,0]]
Y_test_2=[Y_test[:,1],Y_test[:,0]]

def Convolution(input_tensor,filters):
    
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x= Activation('relu')(x)

    return x

def model(input_shape):
    inputs = Input((input_shape))
  
    conv_1= Convolution(inputs,32)
    maxp_1 = MaxPooling2D(pool_size = (2,2)) (conv_1)
    conv_2 = Convolution(maxp_1,64)
    maxp_2 = MaxPooling2D(pool_size = (2, 2)) (conv_2)
    conv_3 = Convolution(maxp_2,128)
    maxp_3 = MaxPooling2D(pool_size = (2, 2)) (conv_3)
    conv_4 = Convolution(maxp_3,256)
    maxp_4 = MaxPooling2D(pool_size = (2, 2)) (conv_4)
    flatten= Flatten() (maxp_4)
    dense_1= Dense(64,activation='relu')(flatten)
    dense_2= Dense(64,activation='relu')(flatten)
    drop_1=Dropout(0.2)(dense_1)
    drop_2=Dropout(0.2)(dense_2)
    output_1= Dense(1,activation="sigmoid",name='sex_out')(drop_1)
    output_2= Dense(1,activation="relu",name='age_out')(drop_2)
    model = Model(inputs=[inputs], outputs=[output_1,output_2])
    model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam", metrics=["accuracy"])
  
    return model


Model=model((48,48,3))
Model.summary()

fle_s='Age_sex_detection.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
Early_stop=tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True),
callback_list=[checkpointer,Early_stop]

History=Model.fit(X_train,Y_train_2,batch_size=64,validation_data=(X_test,Y_test_2),epochs=500,callbacks=[callback_list])

Model.evaluate(X_test,Y_test_2)
pred=Model.predict(X_test)

i=0
Pred_l=[]
while(i<len(pred[0])):
    Pred_l.append(int(np.round(pred[0][i])))
    i+=1

results = confusion_matrix(Y_test_2[0], Pred_l)
print(results)

def test_image(ind,images_f,images_f_2,Model):  
    #cv2_imshow(images_f[ind])
    cv2.imshow('face', images_f[ind])
    cv2.waitKey(0)
    image_test=images_f_2[ind]
    pred_1=Model.predict(np.array([image_test]))
    #print(pred_1)
    sex_f=['Male','Female']
    age=int(np.round(pred_1[1][0]))
    sex=int(np.round(pred_1[0][0]))
    print("Predicted Age: "+ str(age))
    print("Predicted Sex: "+ sex_f[sex])

test_image(10,face_crop_f,face_crop_f_2,Model)

test_image(11,face_crop_f,face_crop_f_2,Model)

test_image(12,face_crop_f,face_crop_f_2,Model)

test_image(14,face_crop_f,face_crop_f_2,Model)

test_image(15,face_crop_f,face_crop_f_2,Model)


# print('PRED:')
# print(pred)
# print('Y_test:')
# print(Y_test_2)

# acc = log_loss(Y_test_2, pred)
# print(acc)

#EVAL
# plt.plot(History.history['loss'])
# plt.plot(History.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,wspace=0.35)
# plt.show()
# ##GENDER
# plt.plot(History.history['sex_out_accuracy'])
# plt.plot(History.history['val_sex_out_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,wspace=0.35)

# ##AGE
# fig, ax = plt.subplots()
# ax.scatter(Y_test_2[1], pred[1])
# ax.plot([Y_test_2[1].min(),Y_test_2[1].max()], [Y_test_2[1].min(), Y_test_2[1].max()], 'k--', lw=4)
# ax.set_xlabel('Actual Age')
# ax.set_ylabel('Predicted Age')
# plt.show()

print("end")







## Current issues: don't detect all profile pictures and it cases it does, 
## it falsely crops out the ear instead of the face