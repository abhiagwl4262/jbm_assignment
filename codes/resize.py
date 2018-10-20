# from __future__ import print_function
# import keras
import numpy as np
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import scipy.misc
import os
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         # width_shift_range=0.2,
#         # height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         # zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

base_dir = "/home/hitech/Desktop/MNIST/with_keras/"
train_dir = base_dir + "train/"
test_dir = base_dir + "test/"

batch_size = 1
num_classes = 2
epochs = 12
img_rows, img_cols = 960, 1260
input_shape = (img_rows, img_cols, 3)

def read_data(directory_, images, labels):

    postives_folder = directory_ + "positives/" 
    negatives_folder= directory_ + "negatives/"
    positive_images = [image for image in os.listdir(postives_folder)]
    negative_images = [image for image in os.listdir(negatives_folder)]
    for pos in positive_images:
    	print(postives_folder + pos)
        x = scipy.misc.imread(postives_folder + pos) 
        # x = img_to_array(x)
        x = scipy.misc.imresize(x, (img_rows, img_cols, 3))
        scipy.misc.imsave(postives_folder + pos, x)
        print(x.shape)        
        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, img_rows, img_cols)
        # labels.append(1)
        # images.append(x)

    for neg in negative_images:
    	print(negatives_folder + neg)      
        x = scipy.misc.imread(negatives_folder + neg) 
        # x = img_to_array(x)
        x = scipy.misc.imresize(x, (img_rows, img_cols, 3))
        scipy.misc.imsave(negatives_folder + neg, x)
        print(x.shape)        

        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, img_rows, img_cols)
        # labels.append(0)
        # images.append(x)

    # num_images = len(images)
    # print(num_images)
    # images = np.array(images)
    # if K.image_data_format() == 'channels_first':
    #     images = images.reshape(num_images, 3, img_rows, img_cols)
    #     input_shape = (3, img_rows, img_cols)
    # else:
    #     images = images.reshape(num_images, img_rows, img_cols, 3)    
    #     input_shape = (img_rows, img_cols, 3)


training_labels = []
training_images = []
testing_labels = []
testing_images = []

read_data(train_dir, training_images, training_labels)
read_data(test_dir, testing_images, testing_labels)

# training_labels = keras.utils.to_categorical(training_labels, num_classes)
# testing_labels = keras.utils.to_categorical(testing_labels, num_classes)

# training_images = np.array(training_images)
# training_labels = np.array(training_labels)
# testing_images  = np.array(testing_images)
# testing_labels  = np.array(testing_labels)

# print ("training_images are ", training_images.shape)
# print ("testing_images are  ", testing_images.shape)
# print ("training_labels are ", training_labels.shape)
# print ("testing_labels are  ", testing_labels.shape)



