'''
main file

Usage : python3 main.py Datapath
'''
import Datapreprocess
from CAE import Autoencoder
import os
import numpy as np
import sys


#load data set
filepath = sys.argv[1]

rgb_data , gray_data = Datapreprocess.getData(filepath)
print("RGB data:{},Gray data:{}".format(np.shape(rgb_data),np.shape(gray_data)))

#train models
autoencoder = Autoencoder()
autoencoder.train(gray_data ,rgb_data, gray_data,rgb_data , 16,10)
decoder_image = autoencoder.getDecodedImage(gray_data)

#visualization
plt.figure(figsize = (20,4))

for i in range(10):
    #orignal
    subplot = plt.subplot(2,10,i+1)
    plt.imshow(Y_train[i].reshape(28,28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
    #reconstructed image
    subplot = plt.subplot(2,10,i+11)
    plt.imshow(decoder_image[i].reshape(28,28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
