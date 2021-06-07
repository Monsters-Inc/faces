from data import load_data
import sys
from gender_model import train_gender_model
from age_model import train_age_model

# Settings
image_folder = "../dataset"
dataset = "../data/full_dataset.csv"
gender_model_save = 'models/gender_final.h5'
age_model_save = 'models/age_final.h5'

img_shape = (96, 96, 1)
no_images = 0
gender_equal = False
test_size = 0.25
require_age = False 
preprocessing = ['he']
logging = True 

batch_size = 64
epochs = 300
monitor = 'val_loss'
type_m = 'B'

if len(sys.argv) > 1:
    type_m = sys.argv[1].upper()

if type_m != 'G':
  require_age = True


train_images, test_images, train_labels, test_labels = load_data(dataset, 
                                                                  image_folder, img_shape, test_size, 
                                                                  no_images = 0,
                                                                  equal = gender_equal,
                                                                  require_age = require_age,
                                                                  preprocessing = preprocessing, 
                                                                  logging = logging)

def gender():
    train_labels_gender = train_labels[['M', 'K']].values
    test_labels_gender = test_labels[['M', 'K']].values

    print(train_images.shape)
    model = train_gender_model(train_images, test_images, train_labels_gender, test_labels_gender, img_shape, 
                               batch_size = batch_size, 
                               epochs = epochs, 
                               model_save = gender_model_save, 
                               monitor = monitor)

    score = model.evaluate(test_images, test_labels_gender)[1]

    print('Gender test accuracy: ', score)

    return model


def age():

    train_labels_age= train_labels['age'].values
    test_labels_age= test_labels['age'].values
    print(train_labels_age)

    model = train_age_model(train_images, test_images, train_labels_age, test_labels_age, img_shape, 
                               batch_size = batch_size, 
                               epochs = epochs, 
                               model_save = age_model_save, 
                               monitor = monitor)



    score = model.evaluate(test_images, test_labels_age)
    print('Mean age error: ', score)

    return model

if type_m == 'G':
    gender_model = gender()

elif type_m == 'A':
    age_model = age()

elif type_m == 'B':
    gender_model = gender()
    age_model = age()


else:
    print('Usage: python run_model.py {A/G}')
    quit()