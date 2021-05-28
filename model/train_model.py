import sys
from gender_model import train_gender_model
from age_model import train_age_model
from data import data_age, data_gender
import numpy as np

# Settings
image_folder = "grayscale"
full_dataset_folder = "../dataset/"
dataset = "../data/full_dataset.csv"
test_size = 0.25
img_shape = (96, 96, 1)
logging = False
gender_model_save = 'g_final.h5'
age_model_save = 'a_final.h5'
batch_size = 64
epochs = 300
monitor = 'val_loss'
augumentation = False
preprocessing = ['he']

type_m = 'B'

if len(sys.argv) > 1:
    type_m = sys.argv[1].upper()

    #if len(sys.argv) > 2:
        #preprocessing.append(sys.argv[2].lower())
        #gender_model_save = 'g_testing_'+sys.argv[2].lower()+'.h5'

# train gender model
def gender():
    X_train, X_test, y_train, y_test = data_gender(dataset, image_folder, img_shape, test_size, preprocessing, logging)
    return X_test, y_test, train_gender_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, augumentation, gender_model_save, monitor)
# Train age model


def age(equal = False):
    X_train, X_test, y_train, y_test = data_age(dataset, full_dataset_folder, equal, img_shape, test_size, preprocessing, logging)
    return X_test, y_test, train_age_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, augumentation, age_model_save, monitor)
        

if type_m == 'G':
    X_test, y_test, gender_model = gender()
    score = gender_model.evaluate(X_test, y_test)[1]
    print('Gender test accuracy: ', score)
    # GÖR VAFAN DU VILL HÄR :)

elif type_m == 'A':
    X_test, y_test, age_model = age()
    score = age_model.evaluate(X_test, y_test)[1]
    print('Age test accuracy: ', score)

elif type_m == 'B':
    X_test_gender, y_test_gender, gender_model = gender()
    X_test_age, y_test_age,  age_model = age()

    gender_score = gender_model.evaluate(X_test_gender, y_test_gender)[1]
    age_score = age_model.evaluate(X_test_gender, y_test_gender)[1]
    print('Mean Age Error: ', age_score)
    print('Gender accuracy: ', gender_score)

    score = gender_score

    # GÖR VAFAN DU VILL HÄR :)

else:
    print('Usage: python run_model.py {A/G}')
    quit()

# This appends the accuracy to results.txt
f = open("results_age_validation_dataset.txt", "a")
f.write(str(score)+'\n')
f.close()
