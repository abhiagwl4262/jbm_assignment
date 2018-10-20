
# coding: utf-8

# In[2]:


import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join

from keras.models import Model
from keras.layers import Input, Merge


# In[3]:


batch_size = 16
img_width, img_height = 256, 256
num_classes = 2
n_epoches = 50
input_shape = (img_height, img_width, 3)


# In[4]:


train_data_dir = "../data/train"
valid_data_dir = "../data/valid"


# In[5]:


train_datagen = ImageDataGenerator(rescale=1./255, 
#                                featurewise_center=False,  # set input mean to 0 over the dataset
#                                samplewise_center=True,  # set each sample mean to 0
#                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
#                                samplewise_std_normalization=True,  # divide each input by its std
#                                 zca_whitening=False,  # apply ZCA whitening
#                                 rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#                                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#                                horizontal_flip=True,  # randomly flip images
#                                vertical_flip=True # randomly flip images                                
)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    seed=10)

val_datagen = ImageDataGenerator(rescale=1. / 255)

valid_generator = val_datagen.flow_from_directory(valid_data_dir,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size,
                                                seed=10)

print ("train data: ", train_generator.n)
print ("valid data: ", valid_generator.n)


# In[9]:


def create_baseline():
    # create model  
    model = Sequential()
    BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape, padding = 'same'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(num_classes, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(num_classes, activation='softmax'))
    return model

# In[47]:

base_model = create_baseline()
base_model.compile(Adam(lr=.001, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


base_model.save_weights("jbm_custom_weights_15.h5")


# In[48]:


base_model.fit_generator(train_generator,
                      steps_per_epoch = train_generator.n // batch_size,
                      validation_data = valid_generator,
                      validation_steps = valid_generator.n // batch_size,
                      epochs = n_epoches,
                      verbose = 1)



base_model.save_weights("jbm_custom_smallNet_50.h5")

