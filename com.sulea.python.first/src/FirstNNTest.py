'''
Created on Oct 26, 2016

@author: sulea
'''
inputset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

from keras.models import Sequential
model = Sequential() 

from keras.layers import Dense, Activation
# 1st layer
# Dense: 2 inputs, 1 outputs 
model.add(Dense(output_dim=1, input_dim=2))
# Activation function: rectified linear unit
model.add(Activation("relu"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

X_train = [[-6,1],[3,-5]]


Y_train = [-1,1]

model.fit(X_train, Y_train, nb_epoch=5, batch_size=10)
