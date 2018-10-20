from __future__ import print_function
import keras
import os
import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_augmentation = False

base_dir = "/home/abhishek/with_keras/"
train_dir = base_dir + "train/"
test_dir  = base_dir  + "test/"

batch_size = 32
num_classes = 2
epochs = 10000
img_rows, img_cols = 256, 256
input_shape = (img_rows, img_cols, 3)

def read_data(directory_, images, labels):

    postives_folder = directory_ + "positives/"
    negatives_folder= directory_ + "negatives/"
    positive_images = [image for image in os.listdir(postives_folder)]
    negative_images = [image for image in os.listdir(negatives_folder)]
    for pos in positive_images:
        x = load_img(postives_folder + pos) 
        x = img_to_array(x)
        x = np.resize(x, (img_rows, img_cols, 3))
        labels.append(1)
        images.append(x)

    for neg in negative_images:
        x = load_img(negatives_folder + neg) 
        x = img_to_array(x)
        x = np.resize(x, (img_rows, img_cols, 3))
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


def inception_block(num_kernels, input_img):
    tower_1 = Conv2D(num_kernels, (1,1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(num_kernels, (3,3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(num_kernels, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(num_kernels, (5,5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((2,2), strides=(1,1), padding='same')(input_img)
    tower_3 = Conv2D(num_kernels, (1,1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
    return output


def create_baseline():

    # create model  
    model = Sequential()
    BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
    input_img = Input(shape = input_shape)
    block1_out  = inception_block(32, input_img)
    block2_out  = inception_block(64, block1_out)
    
    output = Conv2D(256, (1,1), padding='same', activation='relu')(block2_out)
    output = Flatten()(output)
    # Dense_out = Dense(128, activation='softmax')(output)
    Dense_out = Dense(num_classes, activation='softmax')(output)
    model = Model(inputs = input_img, outputs = Dense_out)
    print(model.summary())
    return model

    # model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape, padding = 'same'))
    # # model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',  padding = 'same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    # # model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    # model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding = 'same'))
    # # model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding = 'same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    # model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    # # model.add(Conv2D(256, kernel_size=(3, 3),activation='relu', padding = 'same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    # # model.add(Conv2D(512, (3, 3), activation='relu'))
    # # model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))

    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    # return model


model_path = './defect_classifier_smallNet_depthConv.h5'
# prepare callbacks
callbacks = [
    # EarlyStopping(
    #     monitor='loss', 
    #     patience=10,
    #     mode='max',
    #     verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]

model = create_baseline()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(lr=0.001, decay=1e-5),
              metrics=['accuracy'])

if data_augmentation:
    train_datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True) # randomly flip images

    train_generator = train_datagen.flow(
            training_images, training_labels,
            batch_size=batch_size
            )

    model.fit_generator(
            train_generator,
            steps_per_epoch=25, # batch_size,
            epochs=epochs,
            shuffle = True,
            validation_data=(testing_images, testing_labels),
            )


if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(training_images, training_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle = True,
              validation_data=(testing_images, testing_labels),
              callbacks=callbacks
              )
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()