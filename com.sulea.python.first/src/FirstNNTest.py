'''
Created on Oct 26, 2016

@author: sulea
'''

import numpy as np
import matplotlib.pyplot as plt

# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

# load and transpose input data set
inputset = np.loadtxt("learningsets/hyperplane2d_INPUTS.csv", delimiter=",")
inputset = inputset.transpose()

# load output data set
outputset = np.loadtxt("learningsets/hyperplane2d_OUTPUTS.csv", delimiter=",")
outputset = outputset

# plot the training set
plt.scatter(inputset[:,0], inputset[:,1], s=40, c=outputset, cmap=plt.get_cmap('Spectral'))
plt.show()

# build the simple perceptron
from keras.models import Sequential
model = Sequential() 

from keras.layers import Dense
# The neuron
# Dense: 2 inputs, 1 outputs . Linear activation
model.add(Dense(output_dim=1, input_dim=2, activation="linear"))
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='sgd')
model.fit(inputset, outputset, nb_epoch=100, batch_size=10)

# evaluate the model
scores = model.evaluate(inputset, outputset)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate prediction for one input
testset = np.array([[5],[1]])
predictions = model.predict(testset.transpose())

# round predictions
rounded = [round(x) for x in predictions]
print(rounded)