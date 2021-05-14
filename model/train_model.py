import sys
from gender_model import train_gender_model
from age_model import train_age_model
from data import data
import numpy as np

# Settings
image_folder = "he_equal_distribution_pictures"
full_dataset_folder = "../dataset"
dataset = "../data/full_dataset.csv"
test_size = 0.25
img_shape = (96, 96, 1)
logging = False
gender_model_save = 'g_bw_final.h5'
age_model_save = 'a_final.h5'
batch_size = 64
epochs = 300
multiple_runs = True
monitor = 'val_loss'
augumentation = False

# train gender model
def gender():
    X_train, X_test, y_train, y_test = data(dataset, image_folder, img_shape, test_size, augumentation, logging)
    return X_test, y_test, train_gender_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, gender_model_save, monitor)
# Train age model


def age(equal = False):
    X_train, X_test, y_train, y_test = data(dataset, full_dataset_folder, equal, img_shape, test_size, augumentation, logging)
    return X_test, y_test, train_age_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, age_model_save, monitor)

type_m = 'B'

if len(sys.argv) > 1:
    type_m = sys.argv[1].upper()

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


# Train model

# This appends the accuracy to results.txt if doing multiple runs
if multiple_runs:
    f = open("results.txt", "a")
    f.write(str(score)+'\n')
    f.close()
