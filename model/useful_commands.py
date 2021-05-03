
from preprocessing_tools import resize_images, age_gender_division

age_gender_division('../dataset', 'equal_distribution_pictures', '../data/full_dataset.csv', '../data/equal_distribution.csv', True)

resize_images('equal_distribution_pictures', 'resized_equal_distribution_pictures', (48, 48), True)

