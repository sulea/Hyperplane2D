'''
Created on Feb 18, 2017

@author: sulea
'''
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# function to filter cars (1) and cats (3)
def collectCatsAndCars(X_train_all, Y_train_all):
    X_train = []
    Y_train = []
    i = 0
    for y in Y_train_all:
        if y == 1 or y == 3:
            X_train.append(X_train_all[i])
            Y_train.append(y)
        i = i + 1
    
    return np.array(X_train), np.array(Y_train)

# load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# filter the whole dataset and get only cats and cars
(X_train, Y_train) = collectCatsAndCars(X_train, Y_train)
(X_test, Y_test) = collectCatsAndCars(X_test, Y_test)

# create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i])
# show the plot
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# filter the whole dataset and get only cats and cars
(X_train, Y_train) = collectCatsAndCars(X_train, Y_train)
(X_test, Y_test) = collectCatsAndCars(X_test, Y_test)

# create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i])
# show the plot
plt.show()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
