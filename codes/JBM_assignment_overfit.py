from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datagen = ImageDataGenerator(
        rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

base_dir = "/home/abhishek/with_keras/"
train_dir = base_dir + "train/"
test_dir = base_dir + "test/"

batch_size = 32
num_classes = 2
epochs = 1000
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)

def read_data(directory_, images, labels):

    postives_folder = directory_ + "positives/" 
    negatives_folder= directory_ + "negatives/"
    positive_images = [image for image in os.listdir(postives_folder)]
    negative_images = [image for image in os.listdir(negatives_folder)]
    for pos in positive_images:
        # print(postives_folder + pos)
        x = load_img(postives_folder + pos) 
        x = img_to_array(x)
        x = np.resize(x, (img_rows, img_cols, 3))
        # print(x.shape)
        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, img_rows, img_cols)
        labels.append(1)
        images.append(x)

    for neg in negative_images:
        x = load_img(negatives_folder + neg) 
        x = img_to_array(x)
        x = np.resize(x, (img_rows, img_cols, 3))
        # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, img_rows, img_cols)
        labels.append(0)
        images.append(x)

    num_images = len(images)
    print(num_images)
    images = np.array(images)
    if K.image_data_format() == 'channels_first':
        images = images.reshape(num_images, 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        images = images.reshape(num_images, img_rows, img_cols, 3)    
        input_shape = (img_rows, img_cols, 3)


training_labels = []
training_images = []
testing_labels = []
testing_images = []

read_data(train_dir, training_images, training_labels)
read_data(test_dir, testing_images, testing_labels)

training_labels = keras.utils.to_categorical(training_labels, num_classes)
testing_labels = keras.utils.to_categorical(testing_labels, num_classes)

training_images = np.array(training_images)
training_labels = np.array(training_labels)
testing_images  = np.array(testing_images)
testing_labels  = np.array(testing_labels)

print ("training_images are ", training_images.shape)
print ("testing_images are  ", testing_images.shape)
print ("training_labels are ", training_labels.shape)
print ("testing_labels are  ", testing_labels.shape)



# for i in range(0, 6):
#   plt.imshow(training_images[i])
#   plt.show()

# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# for X_batch, y_batch in datagen.flow(training_images, training_labels, batch_size=batch_size):
#   # create a grid of 3x3 images
#   for i in range(0, batch_size):
#       # pyplot.subplot(330 + 1 + i)
#       plt.imshow(X_batch[i])
#       # show the plot
#       plt.show()
#   break

def create_baseline():
    # create model  
    model = Sequential()
    BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape, padding = 'same'))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',  padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = create_baseline()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(lr=0.001, decay=1e-5),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True
        )

train_generator = train_datagen.flow(
        training_images, training_labels,
        batch_size=batch_size
        )

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=25, # batch_size,
#         epochs=epochs,
#         shuffle = True,
#         # validation_data=validation_generator,
#         # validation_steps=800 // batch_size
#         )

# # # model.save_weights('first_try.h5')  # always save your weights after training or during training

model.fit(training_images, training_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle = True,
          validation_data=(testing_images, testing_labels)
          )
# # score = model.evaluate(x_test, y_test, verbose=0)
# # print('Test loss:', score[0])
# # print('Test accuracy:', score[1])
