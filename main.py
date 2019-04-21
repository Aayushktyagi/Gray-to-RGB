'''
main file

Usage : python3 main.py Datapath
'''
import Datapreprocess
from CAE import Autoencoder
import os
import numpy as np
import sys
import matplotlib.pyplot as plt


#load data set
filepath = sys.argv[1]

rgb_data , gray_data = Datapreprocess.getData(filepath)
print("RGB data:{},Gray data:{}".format(np.shape(rgb_data),np.shape(gray_data)))
#normalize data
rgb_data = rgb_data / 255.0
gray_data =gray_data / 255.0
print("RGB data:{},Gray data:{}".format(np.shape(rgb_data),np.shape(gray_data)))


model_filepath = '/media/aayush/Work/Work/Github/Generative_models/ColoriseImage/weights/cp-{epoch:04d}.ckpt'


#train models
autoencoder = Autoencoder()
autoencoder.train(gray_data ,rgb_data, gray_data,rgb_data ,16 ,200,model_filepath)
decoder_image = autoencoder.getDecodedImage(gray_data)
print(np.shape(decoder_image))

#visualization

autoencoder.showLoss()
plt.figure(figsize = (20,4))

for i in range(10):
    #orignal
    subplot = plt.subplot(3,10,i+1)
    plt.imshow(gray_data[i].reshape(224,224))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
    #reconstructed image
    subplot = plt.subplot(3,10,i+11)
    plt.imshow(rgb_data[i].reshape(224,224,3))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
    subplot = plt.subplot(3,10,i+21)
    plt.imshow(decoder_image[i].reshape(224,224,3))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
