import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from model import train_model
from data import data, data_final

# Settings
image_folder = "resized_96_equal_distribution_pictures/"
dataset = "../data/full_dataset.csv"
test_size = 0.25
img_shape = (96, 96, 3)
logging = False
model_save = 'final.h5'
batch_size = 64
epochs = 500

X_train, X_test, y_train, y_test = data_final(dataset, image_folder, img_shape, test_size, logging)

# Train a new model
model = train_model(X_train, X_test, y_train, y_test, img_shape, batch_size, epochs, model_save)

print('Evaluating model: ')
model.evaluate(X_test, y_test)

print('Predicting: ')
pred = model.predict(X_test)

pred_women = []
pred_men = []

for i in range(len(pred[0])):
    pred_women.append(int(np.round(pred[0][i])))
    pred_men.append(int(np.round(pred[1][i])))

results_men = confusion_matrix(y_test[1], pred_men)
print(results_men)

results_women = confusion_matrix(y_test[0], pred_women)
print(results_women)

report_men = classification_report(y_test[1], pred_men)
print('Report men: ')
print(report_men)

report_women = classification_report(y_test[0], pred_women)
print('Report women: ')
print(report_women)

