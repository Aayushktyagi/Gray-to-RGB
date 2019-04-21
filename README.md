# Image Colorization using Autoencoders

Convert an gray scale image into rgb image using Convolutional Autoencoder
## How to use
Download the flower dataset from [here](https://www.kaggle.com/olgabelitskaya/flower-color-images).Dataset contains 210 flower images.

- Extract file :
> tar -xvf filename

> mv ./filename/flower_images ../../
- Training
> python main.py

## Network Architecture

![Architecture](https://github.com/Aayushktyagi/Grey-to-RGB/blob/master/Results/Network_image_colourize.png)

## Results
### Results after 3000 epochs 
![After 3000 epochs](https://github.com/Aayushktyagi/Gray-to-RGB/blob/master/Results/Results_e_3000.png)

### Results after 6000 epochs
![After 6000 epochs](https://github.com/Aayushktyagi/Gray-to-RGB/blob/master/Results/Results_e_6000.png)
