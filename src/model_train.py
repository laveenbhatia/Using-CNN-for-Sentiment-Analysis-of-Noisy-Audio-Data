import time
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.filters import sobel


# This function will extract the sobel features of an image return the sobel image


def ext_sobel_features(image):
    sobel_image = sobel(image)
    return sobel_image


# This function will preprocess the training data
# -----------------------------------------------
# Arguments:
#   path - this is the path of the folder, relative to this file which contains training images
def preprocess_training_data(path):
    # The following piece of code is used to normalize the images and augment the training data
    # All the parameters can changed as per requirement
    train_batches = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1,
        # This is the name of the preprocessing function defined above which extracts the sobel features
        preprocessing_function=ext_sobel_features
    )

    # Create training data from train_bathes
    training_data = train_batches.flow_from_directory(
        directory=path,
        target_size=(128, 128),
        subset='training'
    )

    # Create validation data from train_batches
    validation_data = train_batches.flow_from_directory(
        directory=path,
        target_size=(128, 128),
        subset='validation'
    )
    return training_data, validation_data


# This function is used to define the model
# Here we can play around with following hyper-parameters:
# 1. Add or delete layers
# 2. Change number of neurons for each layer
# 3. Change the activation function
def create_model(output_neurons):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_neurons, activation='softmax'))
    model.build((None, 128, 128, 3))
    return model


# This function will compile and fit the model
def train_model(model):
    # Compiling the model
    # Here we can change following hyper-parameters:
    #   1. optimizers
    #   2. loss function
    #   3. metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fitting the model
    # Here we can change following hyper-parameters:
    #   1. number of epochs
    #   2. batch size
    model.fit(train_data,
              epochs=2,
              batch_size=32,
              validation_data=valid_data,
              callbacks=[tensorboard])
    return model


# This variable stores the name of the folder with timestamp which will store the logs of the model
NAME = f'MNIST-CNN-{int(time.time())}'

# Used to store logs
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

# Path of the folder, relative to this file where training spectrogram images are stored
TRAINING_IMAGES_DIRECTORY = '../data/Spectrogram/Training'
categories = os.listdir(TRAINING_IMAGES_DIRECTORY)
num_classes = len(categories)

# Creating the training and validation data for our model
train_data, valid_data = preprocess_training_data(TRAINING_IMAGES_DIRECTORY)

class_dict = train_data.class_indices
print(class_dict)

# Creating the model
model = create_model(output_neurons=num_classes)
print(model.summary())

# Training the model
model = train_model(model)

# Saving the model
model.save('model_cnn')

