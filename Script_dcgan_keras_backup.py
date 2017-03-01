############
## This script is to backup my implementation of DCGAN in keras, which is not actually working.
## Still need some effort to debug to find why. 
## The batch normalization in the discriminator screws the results. 
## The generated images and the real images batch should be feed to the discriminator as different minibatch
## not the same batch, so that the mean and std in BN step are different for these two different batches, 
## but just do one time gradient updating. No way to implement in Keras.
## --- Bin Jin, 11:55 am, July 26th, 2016

from __future__ import division
import sys
sys.path.insert(0, '../utils/')

%matplotlib inline
import matplotlib.pyplot as plt
import cv2, time, os, h5py
from scipy.io import savemat, loadmat
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import initializations
from keras.utils import np_utils, generic_utils
from tqdm import tqdm
import h5py

#########
##parameters
#########
k = 1     # # of iterations for training a discriminator before training generator

weight_decay = 1e-5         # l2 weight decay
lr = 0.0001       # initial learning rate for adam
b1 = 0.5          # momentum term of adam
batchsize = 128      # # of examples in batch
nepoch = 5        # # of epochs for training
    
nz = 100          # # of dim for Z
nc = 3            # # of channels in image
npx = 64          # # of pixels width/height of images
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer

########
## Initialization methods, to be the same as in theano implementation
########
def my_init1(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)

def my_init2(shape, name=None):
    value = np.random.normal(loc=1.0, scale=0.02, size=shape)
    return K.variable(value, name=name)

########
## Generator model
########
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=nz, output_dim=ngf*8*4*4,W_regularizer=l2(weight_decay),init=my_init1,bias=False))
    model.add(BatchNormalization(mode=1,gamma_init=my_init2)) # no weight decay on the parameters of this layer
    model.add(Activation('relu'))
    model.add(Reshape((ngf*8, 4, 4)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(ngf*4, 5, 5, border_mode='same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(ngf*2, 5, 5, border_mode='same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(ngf, 5, 5, border_mode='same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))    
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nc, 5, 5, border_mode='same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(Activation('tanh'))
    return model


########
## Discriminator model
########
def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(ndf, 5, 5, subsample=(2, 2), input_shape=(nc, npx, npx), border_mode = 'same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))
    model.add(Convolution2D(ndf*2, 5, 5, subsample=(2, 2), border_mode = 'same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(ndf*4, 5, 5, subsample=(2, 2), border_mode = 'same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(ndf*8, 5, 5, subsample=(2, 2), border_mode = 'same',W_regularizer=l2(weight_decay),init=my_init1))
    model.add(BatchNormalization(mode=2, axis=1,gamma_init=my_init2))    
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1,W_regularizer=l2(weight_decay),init=my_init1,bias=False))
    model.add(Activation('sigmoid'))
    return model


########
## Combine generator with discriminator, but do not train discriminator in between
########
def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

########
## Visualize an image batch
########
def visualizeimages(images,figname=None):
    images = (np.rint((images+1.)*127.5)).astype(np.uint8)
    
    fig = plt.figure(figsize=(20,20))
    sz = images.shape
    figlen = np.floor(np.sqrt(sz[0]))
    for ii in range(int(figlen*figlen)):
        ax = plt.subplot(figlen, figlen, ii+1)
        img = images[ii]
        img = np.transpose(img,[1,2,0])
        plt.imshow(img)
        ax.axis('off')
    
    if figname is not None:
        plt.savefig(figname,format='pdf')

    # plt.show()
    plt.close()


########
## Build and compile the keras models
########
# lrt = K.variable(value=lr)
myop = Adam(lr=lr,beta_1=b1)
G = generator_model()
G.compile(loss='mse',optimizer=myop)
D = discriminator_model()
D.compile(loss='binary_crossentropy',optimizer=myop, metrics=["accuracy"])

# x = Input(shape=((nz,)))
# y = G(x)
# D.trainable = False
# z = D(y)
# G_D = Model(x, z)
G_D = generator_containing_discriminator(G,D)
G_D.compile(loss='binary_crossentropy',optimizer=myop, metrics=["accuracy"])


########
## Train generator and discriminator
########
datagen = ImageDataGenerator()
path_results = '/iccluster_data/bjin/Dcgan/results/Pro/'
## start training
for epoch in range(nepoch):
    batch_count = 0
    print 'epoch: ', epoch + 1
    
    progbar = generic_utils.Progbar(np.floor(len(images)/batchsize) * batchsize)
    for image_batch in datagen.flow(images,batch_size = batchsize):
#     for image_batch in tqdm(datagen.flow(images,batch_size = batchsize), total=np.floor(len(images)/batchsize)):
        batch_count += 1            
        if batch_count > np.floor(len(images)/batchsize):
            break
        
        noise = np.random.uniform(-1,1,(batchsize, nz)).astype(np.float32)
               
        if batch_count % (k+1) == 0:
            generated_images = G.predict_on_batch(noise)
            X_train = np.concatenate((image_batch, generated_images))
            Y_train = np.concatenate((np.ones((batchsize,)),np.zeros((batchsize,))))
            d_loss = D.train_on_batch(X_train, Y_train)
        else:
            g_loss = G_D.train_on_batch(noise,np.ones((batchsize,)))    
        
        progbar.add(image_batch.shape[0], values=[("g_loss", g_loss[0]),('d_loss',d_loss[0])])

        if batch_count % 100 == 0:
            print 'Discriminator loss', d_loss[0], 'Discriminator accuracy', d_loss[1]
            print 'Generator loss', g_loss[0], 'Generator accuracy', g_loss[1]
            print 'visualize generated samples'
            # 'Generating images..'
            generated_images = G.predict_on_batch(noise)
            figname = path_results + str(epoch+1) + '_' + str(batch_count) +'.pdf'
            visualizeimages(generated_images,figname)
            # visualizeimages(image_batch)

