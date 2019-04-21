'''
load rgb images and return dataset containing
rgb and gray scale images in (n,x,y,channel) format
'''


import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

#load datset
def getData(filepath):
  rgb_list = []
  gray_list = []
  # filepath = '/media/aayush/Work/Work/Github/Generative_models/Autoencoders/ImageColourization/flower-color-images'
  file_path = glob.glob(os.path.join(filepath,'*.png'))

  for img_path  in file_path:
    rgb_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.resize(rgb_img,(224,224))
    gray_img = cv2.resize(gray_img,(224,224))
    rgb_list.append(rgb_img)
    gray_list.append(gray_img)

  rgb_images = np.stack(rgb_list,axis=0)
  gray_images = np.stack(gray_list,axis=0)
  gray_images = np.reshape(gray_images,(len(gray_images),224,224,1))

  print(np.shape(rgb_images),np.shape(gray_images))
  return rgb_images , gray_images

#rgb_images , gray_images = getData()
#print(np.shape(rgb_images),np.shape(gray_images))
