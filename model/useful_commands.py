
from preprocessing_tools import resize_images, age_gender_division

resize_images('../full_dataset', 'resized_pictures', (48, 48), True)

age_gender_division('resized_dataset', 'preprocessed_pictures', '../data/full_dataset.csv', True)

