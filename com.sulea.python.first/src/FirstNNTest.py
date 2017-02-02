'''
Created on Oct 26, 2016

@author: sulea
'''

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal as D

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load and transpose input data set
inputset = np.loadtxt("learningsets/hyperplane2d_INPUTS.csv", delimiter=",")
inputset = inputset.transpose()

# load output data set
outputset = np.loadtxt("learningsets/hyperplane2d_OUTPUTS.csv", delimiter=",")
outputset = outputset

# initialize plotter
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
# plot the training set
plt.scatter(inputset[:,0], inputset[:,1], s=40, c=outputset, cmap=plt.get_cmap('Spectral'))
plt.show()

# build the simple perceptron
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization

model = Sequential() 
# The neuron
# Dense: 2 inputs, 1 outputs . Linear activation
model.add(Dense(output_dim=1, input_dim=2, activation="linear"))
model.add(BatchNormalization())
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='sgd')
history = model.fit(inputset, outputset, validation_split=0.2, nb_epoch=50, batch_size=10)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate the model
scores = model.evaluate(inputset, outputset)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate prediction for one input
testset = np.array([[10],[0]])
predictions = model.predict(testset.transpose())
print predictions

# round predictions
rounded = [round(x) for x in predictions]
print(rounded)


# get the weights of the neural network
for layer in model.layers:
    weights = layer.get_weights()
# the math of the prediction
print(weights)

'''

prediction_calc = testset[0]*weights[0][0]+testset[1]*weights[0][1]+1*weights[1]
print prediction_calc
print np.float64(prediction_calc)
'''