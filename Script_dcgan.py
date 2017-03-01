from __future__ import division
import sys
sys.path.insert(0, '../utils/')

import cv2, time, os, h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX

import matplotlib
matplotlib.use('Agg') #(matplotlib.get_backend()) #do this before importing pyplot
import pylab as plt

########
## function to visualize the generated images and save the figure
########
def visualizeimages(images,figname=None,showflag=True):
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
    if showflag:
        plt.show()
    plt.close()


########
## the parameters
########
k = 1            # # of discrim updates for each gen update

nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer

lr = 0.0002       # initial learning rate for adam
b1 = 0.5          # momentum term of adam
weightdecay = 1e-5         # l2 weight decay
batchsize = 128      # # of examples in batch
nepoch = 20        # # of epochs for training


# ########
# ## Load AVA dataset
# ########
# print 'Load AVA dataset'

# path_dataset = '/iccluster_data/bjin/AVA_dataset/'

# if not os.path.isfile(path_dataset + 'AVAdataset_64.hdf5'):
#     print 'generating images'
#     hf = h5py.File(path_dataset + 'AVAdataset_224.hdf5','r')

#     images_train = hf['images_train']
#     scores_train = hf['scores_train']
#     images_val_even = hf['images_test_even']
#     scores_val_even = hf['scores_test_even']
#     images_val_uneven = hf['images_test_uneven']
#     scores_val_uneven = hf['scores_test_uneven']
    
#     images = np.concatenate((images_train,images_val_even,images_val_uneven),axis=0)
#     scores = np.concatenate((scores_train,scores_val_even,scores_val_uneven))
#     hf.close()
    
#     images_new = np.zeros([len(images),3,64,64],dtype=np.float32)
#     for i in range(len(images)):
#         img = images[i]
#         img = np.transpose(img,[1,2,0])
#         img = cv2.resize(img,(64,64))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.transpose(img,[2,0,1]).astype(np.float32)
#         img = img / 127.5 - 1.
#         images_new[i] = img
#     print images_new.shape, np.max(img),np.min(img)
    
#     hf = h5py.File(path_dataset + 'AVAdataset_64.hdf5','w')
#     images_hf = hf.create_dataset("images", (len(images),3,64,64), dtype=np.float32, maxshape=(None,3,64,64))
#     images_hf[...] = images_new
#     hf.close()
    
# hf = h5py.File(path_dataset + 'AVAdataset_64.hdf5','r')
# images = hf['images'][:]
# print images.shape


########
## Load the Pro high Dataset
########
print 'Load Pro dataset'
from joblib import Parallel, delayed
import multiprocessing

def processInput(i):
    if i % 10e3 == 0:
        print i,'/',n_files_Pro,' has been loaded '
        start_time = time.time()
    img = cv2.imread(path_dataset + img_names[i].strip()).astype(np.float32)
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img,[2,0,1])
    img = img / 127.5 - 1.
    images[i] = img


path_dataset = '/iccluster_data/bjin/ProDataset/Images_yfcc100M/'
if not os.path.isfile(path_dataset + 'Prodataset_high_64.hdf5'):
    print 'generating Pro dataset'
    n_files_Pro = int(250e3)
    file_metadata = '/iccluster_data/bjin/ProDataset/Images_yfcc100M/Prodataset_yfcc100M_all_filtered_metadata.mat'
    temp = loadmat(file_metadata)
    img_names = temp['img_names']
    scores = temp['aes_scores']
    
    images = np.zeros([n_files_Pro,3,64,64],dtype=np.float32)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(n_files_Pro))

#     for i in range(n_files_Pro):
#         if i % 10e3 == 0:
#             print i,'/',n_files_Pro,' has been loaded ',time.time()-start_time,' seconds'
#             start_time = time.time()
#         img = cv2.imread(path_dataset + img_names[i].strip()).astype(np.float32)
#         img = cv2.resize(img,(64,64))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.transpose(img,[2,0,1])
#         img = img / 127.5 - 1.
#         images[i] = img
    print images.shape, images.dtype#, np.max(img), np.min(img)

    hf = h5py.File(path_dataset + 'Prodataset_high_64.hdf5','w')
    images_hf = hf.create_dataset("images", (n_files_Pro,3,64,64), dtype=np.float32, maxshape=(None,3,64,64))
    images_hf[...] = images
    hf.close()
    
hf = h5py.File(path_dataset + 'Prodataset_high_64.hdf5','r')
images = hf['images'][:]
print images.shape, np.unique(images[1])


########
## Build DCGAN training and predicting function
########

## pre-defined layers implemented in theano
relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

## initialization methods
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

## weights of the generator model
gw  = gifn((nz, ngf*8*4*4), 'gw')
gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb')
gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4')
gwx = gifn((ngf, nc, 5, 5), 'gwx')

## weights of the discriminator model
dw  = difn((ndf, nc, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf, 5, 5), 'dw2')
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
dw3 = difn((ndf*4, ndf*2, 5, 5), 'dw3')
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
dw4 = difn((ndf*8, ndf*4, 5, 5), 'dw4')
dg4 = gain_ifn((ndf*8), 'dg4')
db4 = bias_ifn((ndf*8), 'db4')
dwy = difn((ndf*8*4*4, 1), 'dwy')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
discrim_params = [dw, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]

## definition of the generator
def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x

## definiton of the discriminator
def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    h4 = T.flatten(h4, 2)
    y = sigmoid(T.dot(h4, wy))
    return y

## define the loss
X = T.tensor4()
Z = T.matrix()

gX = gen(Z, *gen_params)

p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=weightdecay))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=weightdecay))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time.time()
_train_g = theano.function([X, Z], cost, updates=g_updates)
_train_d = theano.function([X, Z], cost, updates=d_updates)
_gen = theano.function([Z], gX)
print '%.2f seconds to compile theano functions'%(time.time()-t)



########
## Train DCGAN
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
            mycost = _train_d(image_batch.astype(np.float32), noise)
        else:
            mycost = _train_g(image_batch.astype(np.float32), noise)
        
        progbar.add(image_batch.shape[0], values=[("g_loss", mycost[0]),('d_loss',mycost[1])])

        if batch_count % 500 == 0:
            print 'loss (g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen):'
            print mycost
            print 'visualize generated samples'
            # 'Generating images..'
            generated_images = _gen(noise)
            figname = path_results + str(epoch+1) + '_' + str(batch_count) +'.pdf'
            visualizeimages(generated_images,figname，showflag=False)
