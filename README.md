# Image Colorization using Autoencoders

Convert an gray scale image into rgb image using Convolutional Autoencoder
## How to use
Download the flower dataset from [here](https://www.kaggle.com/olgabelitskaya/flower-color-images).Dataset contains 210 flower images.

- Extract file :
> tar -xvf filename

> mv ./filename/flower_images ../../

- Training
> python main.py ./flower_images


## Loss Graph
- Graphs depicts Training loss
![Model_loss](https://github.com/Aayushktyagi/Gray-to-RGB/blob/master/Results/Model_loss.png)

## Network Architecture

![Architecture](https://github.com/Aayushktyagi/Grey-to-RGB/blob/master/Results/Network_image_colourize.png)

## Results
### Result after 3000 epochs
- First row contains ground truth gray scale images.
- Second row contains ground truth rgb images.
- Third row contains images generated from the network.
![Results](https://github.com/Aayushktyagi/Gray-to-RGB/blob/master/Results/Results_e_3000.png)

## Conclusion
Convolutional Autoencoder can be used to convert gray scale images into rgb images.Encoder is able
to effectively learn images representaion into latent vector and decoder is able to map learned
image representaion into color image.
