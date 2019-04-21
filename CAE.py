import os
import tensorflow as tf
from tensorflow.keras.layers import Input , Conv2D , UpSampling2D,MaxPooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


class Autoencoder(object):
    '''
    Convolutional Autoencoder to convert gray scale imagas to RGB
    '''

    def __init__(self):

        #Encoder
        input_layer = Input(shape=(224,224,1))
        encoding_conv_layer1 = Conv2D(32,(3,3),activation = 'relu',padding = 'same')(input_layer)
        encoding_conv_layer1 = MaxPooling2D((2,2),padding = 'same')(encoding_conv_layer1)
        encoding_conv_layer2 = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(encoding_conv_layer1)
        encoding_conv_layer2 = MaxPooling2D((2,2),padding = 'same')(encoding_conv_layer2)
        encoding_conv_layer3 = Conv2D(128,(3,3),activation= 'relu',padding='same')(encoding_conv_layer2)
        encoding_conv_layer3 = MaxPooling2D((2,2) , padding = 'same')(encoding_conv_layer3)
        encoding_conv_layer4 = Conv2D(256 , (3,3),activation = 'relu',padding = 'same')(encoding_conv_layer3)
        latent_vector = MaxPooling2D((2,2),padding = 'same')(encoding_conv_layer4)

        #decoder
        decoding_conv_layer1 = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(latent_vector)
        decoding_conv_layer1 = UpSampling2D((2,2))(decoding_conv_layer1)
        decoding_conv_layer2 = Conv2D(64,(3,3),activation ='relu',padding = 'same')(decoding_conv_layer1)
        decoding_conv_layer2 = UpSampling2D((2,2))(decoding_conv_layer2)
        decoding_conv_layer3 = Conv2D(32,(3,3),activation = 'relu',padding ='same')(decoding_conv_layer2)
        decoding_conv_layer3 = UpSampling2D((2,2))(decoding_conv_layer3)
        decoding_conv_layer4 = Conv2D(8,(3,3),activation = 'relu',padding = 'same')(decoding_conv_layer3)
        decoding_conv_layer4 = UpSampling2D((2,2))(decoding_conv_layer4)
        #decoding_conv_layer5 = Conv2D(4,(3,3),activation = 'relu',padding = 'same')(decoding_conv_layer4)
        #decoding_conv_layer5 = UpSampling2D((2,2))(decoding_conv_layer5)
        output_layer = Conv2D(3,(3,3),activation = 'sigmoid',padding = 'same')(decoding_conv_layer4)

        #loss_function=tf.reduce_mean(tf.squared_difference(output_layer , ))
        self._model = Model(input_layer , output_layer)
        self._model.compile(optimizer = 'adam' , loss = 'binary_crossentropy')
        self._model.summary()

    def train(self, input_train,output_train,input_test,output_test,batch_size,epochs,checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,period=5)

        self._model.fit(input_train,
                        output_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_data = (input_test,
                                            output_test),
                        callbacks = [cp_callback])

    def getDecodedImage(self,test_image):
        decoded_image = self._model.predict(test_image)
        return decoded_image

    def showLoss(self):
        plt.plot(self._model.history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
