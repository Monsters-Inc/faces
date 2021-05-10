import os
import cv2
import pandas as pd
import numpy as np

#
# Format image folder name
#
def format_folder_name(folder):
    if not folder.endswith('/'):
        folder = folder+'/'

    return folder

#
# Crops image
#
def image_crop(image, y, x, h, w):
    return image[y:y+h, x:x+w]

#
# Vertical halve images
#
def image_vertical_halver(image, color):
    img = cv2.imread(image, color)

    if color == 0:
        height, width = img.shape
    else:
        height, width, channels = img.shape

    # Top half
    cropped_top = image_crop(img, y=0, x=0, h=int(height/2), w=int(width))
    cropped_bottom = image_crop(img, y=int(height/2), x=0, h=int(height/2), w=int(width))

    return cropped_top, cropped_bottom

#
# Horizontal halve images
#
def image_horizontal_halver(image, color):
    img = cv2.imread(image, color)

    if color == 0:
        height, width = img.shape
    else:
        height, width, channels = img.shape

    cropped_left = image_crop(img, y=0, x=0, h=int(height),w=int(width/2))
    cropped_right = image_crop(img, y=0, x=int(width/2), h=int(height),w=int(width/2))

    return cropped_left, cropped_right

#
# Crops a folder of images vertically and saves them back to destination folder
#
def images_vertical_halver(image_folder, destination_folder, color, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1

            cropped_top, cropped_bottom = image_vertical_halver(image_folder+image, color)
            cv2.imwrite(destination_folder+'top_'+image, cropped_top)
            cv2.imwrite(destination_folder+'bottom_'+image, cropped_bottom)

#
# Crops a folder of images horizontally and saves them back to destination folder
#
def images_horizontal_halver(image_folder, destination_folder, color, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1

            cropped_left, cropped_right = image_horizontal_halver(image_folder+image, color)
            cv2.imwrite(destination_folder+'left_'+image, cropped_left)
            cv2.imwrite(destination_folder+'right_'+image, cropped_right)

#
# Resizes images in image_folder and writes them to destination_folder
#
def resize_images(image_folder, destination_folder, size, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)
    count = 1
    for image in images:
        if logging:
            print('Resizing: '+image+' | '+str(count)+'/'+str(len(images)))
        img = cv2.imread(image_folder+image)
        img = cv2.resize(img, size)
        cv2.imwrite(destination_folder+image, img)
        count+=1

#
# Swap K and F with 1 and M with 0
#
def binarize_gender(df):
    return df['gender'].map({'K':1, 'F':1, 'M':0})

#
# Returns equally distributed dataset, with equal amounts of women as men
#
def equal_distribution_dataset(df):
    df.notna(inplace=True)
    df = binarize_gender(df)
    women_count = df['gender'].value_counts()[1]

    men_df = df[df.gender == 0]
    women_df = df[df.gender == 1]
    men_df = men_df.head(women_count)
    return pd.concat(frames)

#
# Creates folder with equal gender distribution, only people with age in csv
#

def age_gender_division(image_folder, destination_folder, df, path_new_df, logging):
    men_count = 0
    women_count = 0
    df = pd.read_csv(df, sep=';')
    df_ages = df[df['age'].notna()]
    df_ages = df_ages.sample(frac=1)
    women_max_count = df_ages['gender'].value_counts().K

    new_df = pd.DataFrame(columns = ['image', 'gender', 'age'])

    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)
    
    images = os.listdir(image_folder)
    count = 1
    for filename in images:
        if os.path.isfile(image_folder+filename):
            if '.DS_Store' in images:
                images.remove('.DS_Store')
            if logging:
                print('Age and gender dividing: '+filename+' | '+str(count)+'/'+str(len(images)))
        
            if filename in df_ages.values:
                row = df_ages.loc[df_ages['image'] == filename]
                gender = row['gender'].values[0]
                age = row['age'].values[0]

                if (men_count >= women_max_count) and (women_count >= women_max_count):
                    break
                
                elif (gender == 'K' or gender == 'F') and age:
                    img = cv2.imread(image_folder+filename, 1)
                    new_df.append(row)
                    cv2.imwrite(destination_folder+filename, img)
                    women_count += 1
            
                elif gender == 'M' and men_count < women_max_count and age:
                    img = cv2.imread(image_folder+filename, 1)
                    new_df.append(row)
                    cv2.imwrite(destination_folder+filename, img)
                    men_count += 1 
                    
        count+=1
                    
    # Save new df
    new_df.to_csv(path_new_df, sep=';')

#
# Grayscale transform - Preprocess
#
def grayscale(image_folder, destination_folder, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1
            grayscale_img = cv2.imread(image_folder+image, 0)
            cv2.imwrite(destination_folder+image, grayscale_img)

#
# HE transform - Preprocess
#
def he(image_folder, destination_folder, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1
            img = cv2.imread(image_folder+image, 0)
            he_img = cv2.equalizeHist(img)
            cv2.imwrite(destination_folder+image, he_img)

#
# BGR transform - Preprocess
#
def bgr(image_folder, destination_folder, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1
            img = cv2.imread(image_folder+image)
            BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(destination_folder+image, BGR_img)

bgr('resized_96_equal_distribution_pictures', 'resized_96_equal_distribution_pictures_bgr', True)

#
# CLAHE - Preprocess
#
def clahe(image_folder, destination_folder, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1
            img = cv2.imread(image_folder+image)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_l = clahe.apply(l)
            merged_channels = cv2.merge((clahe_l, a, b))
            clahe = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
            cv2.imwrite(destination_folder+image, clahe)

#
# Canny edges - Preprocess
#
def canny_edges(image_folder, destination_folder, logging):
    images = os.listdir(image_folder)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)
    destination_folder = format_folder_name(destination_folder)

    count = 1
    for image in images:
        if os.path.isfile(image_folder+image):
            if logging:
                print(f'Processing: {image} ({count}/{len(images)})')
                count += 1
            img = cv2.imread(image_folder+image, 0)
            height, width = img.shape
            edges = cv2.Canny(img, height, width)
            cv2.imwrite(destination_folder+image, edges)

#
# Convert images to vectors
#
def images_to_vectors(images, image_folder, df, color, logging):
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)

    fnf_count = 0
    result = []
    for image in images:
    	if os.path.isfile(image_folder+image):
    		img = cv2.imread(image_folder+image, color)

    		result.append(img)
    	else:
    		fnf_count += 1
    		df = df[df.image != image]

    if logging:
           print('\nFiles not found: '+str(fnf_count)+'\n')

    return result, df

#
# Returns a list with image names
#
def list_image_names(quantity, image_extension):
    if not image_extension.startswith('.'):
        image_extension = '.'+image_extension

    men = []
    women = []
    for i in range(quantity):
        men.append('m'+str(i+1)+image_extension)
        women.append('w'+str(i+1)+image_extension)

    return men + women

#
# Returns a list with true labels
#
def list_true_labels(quantity):
    men = []
    women = []
    for i in range(quantity):
        men.append(0)
        women.append(1)

    return men + women

#
# Predict images using model
#
def model_predict(model, image_list, image_folder, true_labels, labels, image_dim):
    # Makes sure folders end with '/'
    image_folder = format_folder_name(image_folder)

    resulting_image_list = image_list.copy()

    indices_not_found = []
    images_no_face_found = 0
    test_data = []
    print('Processing images...')
    for i, image in enumerate(image_list):
        grayscale_image = cv2.imread(image_folder+image, 0)

        # Load classifier
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        # Detect faces of varying sizes
        detected_faces = face_cascade.detectMultiScale(grayscale_image)

        if len(detected_faces) > 0:
            col = detected_faces[0][0]
            row = detected_faces[0][1]
            w = detected_faces[0][2]
            h = detected_faces[0][3]
            # Resize image to only be the face
            face_img = grayscale_image[row:row+h, col:col+w]
            img_to_test = cv2.resize(face_img, (image_dim['width'], image_dim['height']))
            img_to_test = img_to_test.reshape(-1, image_dim['width'], image_dim['height'], image_dim['channels'])
            test_data.append(img_to_test)
        else:
            indices_not_found.append(i)
            images_no_face_found += 1

    for index in sorted(indices_not_found, reverse=True):
        del true_labels[index]
        del image_list[index]


    test_data = np.vstack(test_data)
    result = model.predict(test_data)

    # print('\n----- Prediction Finished -----')
    #
    #
    # corrects = 0
    # prediction_count = len(result)
    # for i in range(prediction_count):
    #     pred_index = np.argmax(result[i])
    #     true_index = true_labels[i]
    #     pred_label = labels[pred_index]
    #     true_label = labels[true_index]
    #     print(f'Predicted: {pred_label} | True: {true_label}')
    #
    #     if pred_index == true_index:
    #         corrects += 1
    #     else:
    #         print(f'Image incorrectly predicted: {image_list[i]}')
    #
    # print(f'\nThe model predicted: {corrects}/{prediction_count} correct, whereas {images_no_face_found} images had no faces found')
