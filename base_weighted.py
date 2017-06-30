# coding=utf8
"""
Created on Wed Feb  1 13:09:44 2017

@author: Redouane Lguensat

Eddynet: A convolutional encoder-decoder for the pixel-wise segmentation of AVISO SSH MAPS

(code is based on Keras 2.0, theano dim oredering and tensorflow backend)

"""

############################################# DATA
#---------------------------  segmentation mask
#    0: no eddy
#    1: anticyclonic eddy
#    2: cyclonic eddy
#    3: land or no data

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import measures as m

# Load Segmentation maps training data (from 2000 to 2010)
#SegmaskTot=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/SegmaskTot_2000_2010.npy')
# load SSH AVISO maps data (2011 will be used for validation data)
#SSH_aviso_train=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/SSH_aviso_2000_2011.npy')

###################################################################

# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/h17valen/Deep_learning_pollution/weights/"


###################################################################

from keras.models import Model, load_model
from keras.layers.core import Activation, Reshape, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, merge
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
#from keras.layers.merge.concatenate import concatenate
from keras.utils import np_utils
from keras import backend as K
from keras.layers.merge import concatenate

K.set_image_dim_ordering('th') # Theano dimension ordering in this code

width = 508
height = 508
nbClass = 3

kernel = 5

# Number of max_pooling/upsampling:
depth = 3
# Number of convolution between two max_pooling layers:
nb_conv = 1

def crop():
    c=kernel-1
    for i in range(depth):
        yield c
        c+=(kernel-1)*nb_conv
        c*=2

c=crop()

img_input = Input(shape=(height, width))

######################################ENCODER

x = Reshape((1,height,width))(img_input)

conv1 = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(x)
#x = Dropout(0.1)(conv1)
#x = BatchNormalization()(x)
x = Activation("elu")(conv1) #!!!
x = MaxPooling2D(pool_size=(2, 2))(x)


conv2 = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(x)
x = Dropout(0.1)(conv2)
#x = BatchNormalization()(x)
x = Activation("elu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


conv3 = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(x)
x = Dropout(0.1)(conv3)
#x = BatchNormalization()(x)
x = Activation("elu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(x)
x = Dropout(0.1)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)

#
#######################################DECODER
#

conv3 = Cropping2D(c.next(),data_format="channels_first")(conv3)
up1 = concatenate([UpSampling2D(size=(2, 2))(x), conv3], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up1)
x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)

conv2 = Cropping2D(c.next(),data_format="channels_first")(conv2)
up2 = concatenate([UpSampling2D(size=(2, 2))(x), conv2], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up2)
x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)

conv1 = Cropping2D(c.next(),data_format="channels_first")(conv1)
up3 = concatenate([UpSampling2D(size=(2, 2))(x), conv1], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up3)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)  

####################################### Segmentation Layer

x = Conv2D(nbClass, (1, 1), padding="valid",data_format="channels_first")(x) 
x = Reshape((nbClass, -1))(x) 

#x= Cropping2D(24,data_format="channels_first")(x) 

x = Permute((2, 1))(x)
x = Activation("softmax")(x)
eddynet = Model(img_input, x)
############################################################################################# COMPILE

#print eddynet.summary()

eddynet.compile(optimizer='adadelta', loss='categorical_crossentropy',
                metrics=['accuracy'],
                sample_weight_mode="temporal")


#from keras.utils.vis_utils import plot_model
#plot_model(eddynet,to_file='eddynet.png')

################################################ Train/Test data (PATCH version)

# from guppy import hpy
import h5py as h

# hp=hpy()

# print hp.heap()

hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"

fl=__file__

with h.File(hdf,"a") as f:
    
    if not f.__contains__("masks/training_images/"+fl):
        b=420*420
        a=f["train/Mask"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
        f.create_dataset("masks/train/"+fl,a.shape+(3,),dtype='f4')
        n=2
        l=len(a)/n
        for i in range(n):
            f["masks/train/"+fl][i*l:(i+1)*l]=np_utils.to_categorical(a[i*l:(i+1)*l],3).reshape(a[i*l:(i+1)*l].shape+(3,))
        f.create_dataset("masks/testing_images/"+fl,(0,420*420,3),maxshape=(None,420*420,3),dtype='f4')
        a=f["test/Mask/testing_images/"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
        f["masks/testing_images/"+fl].resize(a.shape+(3,))
        f["masks/testing_images/"+fl][:]=np_utils.to_categorical(a,3).reshape(a.shape+(3,))
        a=f["test/Mask/training_images/"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
        f.create_dataset("masks/training_images/"+fl,(0,420*420,3),maxshape=(None,420*420,3),dtype='f4')
        f["masks/training_images/"+fl].resize(a.shape+(3,))
        f["masks/training_images/"+fl][:]=np_utils.to_categorical(a,3).reshape(a.shape+(3,))

    # Weights
    (a,b,c)=f["masks/train/"+fl].shape
    w=np.full((a,b),1.,dtype=np.float32)
    q=np.argmax(f["masks/train/"+fl][:],axis=-1)
    w[q==1]=100.
    w[q==2]=1000000.
    w=w.reshape(a,-1)
    del q
    f["weights/"+fl][:]=w
    
    # Train nrcs
    train_nrcs=f["train/Nrcs"]#[::]
    # Test nrcs
    test_nrcs=f["test/Nrcs/testing_images"]#[:]

    # Train mask
    train_mask=f["masks/train/"+fl]#[::]
    
    # Test mask
    test_mask=f["masks/testing_images/"+fl]#[:]
    

    batch_size=16
    epochs=6

    eddynet.load_weights(weight_dir+fl)

    #eddynet.fit(train_nrcs,train_mask,shuffle='batch',verbose=1,batch_size=batch_size,epochs=epochs,sample_weight=w)

    eddynet.save_weights(weight_dir+fl)

    #print eddynet.evaluate(test_nrcs,test_mask)

    v=eddynet.predict(test_nrcs)

    f["results/testing_images"+fl]=v
    f["segmentation/testing_images/"+fl]=m.to_classes(v).reshape(-1,420,420)
    
    size_out = 420
    
    test_nrcs=f["test/Nrcs/training_images/"]
    w=eddynet.predict(test_nrcs,verbose=1)

    f.require_dataset("results/training_images/"+fl,shape=(1700,420,420,3),dtype='f4',exact=False)
    f["results/training_images/"+fl][:]=w.reshape(-1,420,420,3) 
    f.require_dataset("segmentation/training_images/"+fl,shape=(1700,420,420),dtype='i8',exact=False)
    f["segmentation/training_images/"+fl][:]=m.to_classes(w).reshape(-1,420,420)

    test_nrcs=f["train/Nrcs/"]
    x=eddynet.predict(test_nrcs,verbose=1)

    f.require_dataset("results/train/"+fl,shape=(nb_train,size_out,size_out,nbClass),dtype='f4',exact=False)
    f["results/train/"+fl][:]=x.reshape(-1,size_out,size_out,nbClass) 
    f.require_dataset("segmentation/train/"+fl,shape=(nb_train,size_out,size_out),dtype='i8',exact=False)
    f["segmentation/train/"+fl][:]=m.to_classes(x).reshape(-1,size_out,size_out)
