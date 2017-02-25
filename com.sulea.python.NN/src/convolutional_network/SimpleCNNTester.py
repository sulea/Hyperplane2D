'''
Created on Feb 21, 2017

@author: sulea
'''
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import cv2
from numpy.core.fromnumeric import argmax

# read own image
#image = cv2.imread('pix/mercedes280.jpg', cv2.IMREAD_UNCHANGED)
image = cv2.imread('pix/audiropika.jpg', cv2.IMREAD_UNCHANGED)

# convert color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resize image
image=cv2.resize(image,(32,32))
# normalize image
image = image.astype('float32')
image = image / 255.0

# get the compiled model
model = load_model('scnn.h5')

# add 0st dimenstion (nb_samples) for input of Convolution2D
reshaped_image = image.reshape( (1,) + image.shape )  

# predict
prediction = argmax(model.predict(reshaped_image))

# show image
plt.imshow(image)
plt.title('This is a '+['car', 'cat'][prediction])
plt.show()