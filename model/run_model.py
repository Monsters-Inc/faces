from model import train_model
from data import data

# Settings
image_folder = "resized_96_equal_distribution_pictures"
dataset = "../data/full_dataset.csv"
test_size = 0.25
img_shape = (96, 96, 1)
logging = False
model_save = 'final.h5'
batch_size = 64
epochs = 500
multiple_runs = True

# Get training and test data
X_train, X_test, y_train, y_test = data(dataset, image_folder, img_shape, test_size, logging)

# Train model
model = train_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, model_save)

# Evaluating model
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])

# This appends the accuracy to results.txt if doing multiple runs
if multiple_runs:
    f = open("results.txt", "a")
    f.write(str(score[1])+'\n')
    f.close()



