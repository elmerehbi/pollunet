# coding=utf8

###########################
# Imports
###########################

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, load_model
from keras.layers.core import Activation, Reshape, Permute
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, merge
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.layers.merge import concatenate
from keras.optimizers import *
import h5py as h
import sys
sys.path.append('../')
import measures as m
import os.path 

###########################

K.set_image_dim_ordering('th') # Theano dimension ordering in this code

# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/h17valen/Deep_learning_pollution/weights/"

# Poids des différentes classes pour le loss
weight_pollution = 500.
weight_land = 5.
weight_boats = 1000.
weight_sea = 1
reload_weights = True

# Network parameters:
width = 508
height = 508
nbClass = 4
kernel = 5
depth = 3
nb_conv = 1
nb_conv_out = 1
activation = "relu"
dropout_down = 0.1
dropout_up = 0

channels_max = 32

def channels(depth):
    return channels_max
#    return channels_max / 2**depth

# Training parameters
training_size = 2000
batch_size=24
epochs=4
optimizer=adadelta()


###########################

def crop():
    c=kernel-1
    for i in range(depth):
        yield c
        c+=(kernel-1)*nb_conv
        c*=2


size_out = width - 2 * list(crop())[-1] - (kernel-1) * (nb_conv_out + nb_conv)

c=crop()


def unet_layers(x,depth):
    if depth == 0:
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
        x=Conv2D(channels(depth)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_down)(x)
    else:
        for i in range(nb_conv):
            x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
    if depth>0:
        y=MaxPooling2D(pool_size=(2,2))(x)
        y=unet_layers(y,depth-1)
        y=UpSampling2D(size=(2,2))(y)
        x=Cropping2D(c.next(),data_format='channels_first')(x)
        x=concatenate([x,y],axis=1)
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_up)(x)
        x=Conv2D(channels(depth)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_up)(x)
    return x

###########################

sar_input = Input(shape=(height, width))
mask_input = Input(shape=(height, width, 2))

sar = Reshape((1,height,width))(sar_input)
mask = Permute((3,1,2))(mask_input)


x = concatenate([sar,mask],axis=1)

x = unet_layers(x,depth)

for i in range(nb_conv_out):
    x = Conv2D(nbClass,1,padding='valid',data_format='channels_first')(x)

x = Reshape((nbClass,-1))(x)

x = Permute((2,1))(x)

x = Activation('softmax')(x)

unet = Model([sar_input,mask_input],x)

###########################

#print unet.summary()
# print size_out

unet.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'],
                sample_weight_mode="temporal")


###########################

hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"
fl=__file__
print fl
b=size_out*size_out
o=(width-size_out)/2

with h.File(hdf,"a") as f:
    nb_train = len(f["train/Nrcs"])
    nb_set1 = len(f["test/Nrcs/training_images"])
    nb_set2 = len(f["test/Nrcs/testing_images"]) 


    n=8
    if not f.__contains__("masks/training_images/"+fl):
        a=f["train/Mask"][:]#[:,o:-o,o:-o]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=3
#        a=a.reshape(-1,b)
        f.require_dataset("masks/train/"+fl,a.shape+(nbClass,),dtype='f4')

        l=len(a)/n+1
        for i in range(n):
            f["masks/train/"+fl][i*l:(i+1)*l]=np_utils.to_categorical(a[i*l:(i+1)*l],nbClass).reshape(a[i*l:(i+1)*l].shape+(nbClass,))
        a=f["test/Mask/testing_images/"][:]#[:,o:-o,o:-o]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=3
#        a=a.reshape(-1,b)
#        f["masks/testing_images/"+fl].resize(a.shape+(nbClass,))
        f.require_dataset("masks/testing_images/"+fl,a.shape+(nbClass,),dtype='f4')
        f["masks/testing_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))
        a=f["test/Mask/training_images/"][:]#[:,o:-o,o:-o]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=3
#        a=a.reshape(-1,b)
        f.require_dataset("masks/training_images/"+fl,a.shape+(nbClass,),dtype='f4')
#        f["masks/training_images/"+fl].resize(a.shape+(nbClass,))
        f["masks/training_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))
        
    if not f.__contains__("weights/"+fl) or reload_weights:
        (a,b,d,c)=f["masks/train/"+fl].shape
        w=np.full((a,size_out*size_out),1.,dtype=np.float32)
        l=len(w)/n+1
        for i in range(n):
            w[i*l:(i+1)*l,:]=np.argmax(f["masks/train/"+fl][i*l:(i+1)*l,o:-o,o:-o],axis=-1).reshape((-1,size_out*size_out))
        w[w==1]=weight_land
        w[w==0]=weight_sea
        w[w==2]=weight_pollution
        w[w==3]=weight_boats
        w=w.reshape(a,-1)
        f.require_dataset('weights/'+fl,(nb_train,size_out*size_out),dtype=np.float32,exact=False)
        f["weights/"+fl][:]=w


###########################

    l = m.randl(training_size,nb_train,m.l4_train)

#    print l
     # Weights
    w=f["weights/"+fl][l]
    # Train nrcs
    train_nrcs=f["train/Nrcs"][l]
    # Train mask
    train_mask=f["masks/train/"+fl][l][:,o:-o,o:-o,:].reshape((-1,size_out*size_out,nbClass))
    # Input mask
    input_mask = f["masks/train/"+fl][l][...,[1,3]]
      

    if os.path.isfile(weight_dir+fl):
        unet.load_weights(weight_dir+fl)

###########################


    unet.fit([train_nrcs,input_mask],train_mask,shuffle=True,verbose=1,batch_size=batch_size,epochs=epochs,sample_weight=w)

    unet.save_weights(weight_dir+fl)

#    exit ()
#    print unet.evaluate(test_nrcs,test_mask)

    del input_mask
    del train_mask
    del train_nrcs

    test_nrcs=f["test/Nrcs/testing_images/"]
    input_mask = f["masks/testing_images/"+fl][:][...,[1,3]] 
    v=unet.predict([test_nrcs,input_mask],verbose=1)

    f.require_dataset("results/testing_images/"+fl,shape=(nb_set2,size_out,size_out,nbClass),dtype='f4',exact=False)
    f["results/testing_images/"+fl][:]=v.reshape(-1,size_out,size_out,nbClass)
    f.require_dataset("segmentation/testing_images/"+fl,shape=(nb_set2,size_out,size_out),dtype='i8',exact=False)
    f["segmentation/testing_images/"+fl][:]=m.to_classes(v).reshape(-1,size_out,size_out)

    del input_mask
    del test_nrcs

    test_nrcs=f["test/Nrcs/training_images/"]
    input_mask = f["masks/training_images/"+fl][:][...,[1,3]]#.reshape(-1,size_out,size_out,2)
    w=unet.predict([test_nrcs,input_mask],verbose=1)

    f.require_dataset("results/training_images/"+fl,shape=(nb_set1,size_out,size_out,nbClass),dtype='f4',exact=False)
    f["results/training_images/"+fl][:]=w.reshape(-1,size_out,size_out,nbClass) 
    f.require_dataset("segmentation/training_images/"+fl,shape=(nb_set1,size_out,size_out),dtype='i8',exact=False)
    f["segmentation/training_images/"+fl][:]=m.to_classes(w).reshape(-1,size_out,size_out)

    del input_mask
    del test_nrcs

    # test_nrcs=f["train/Nrcs/"]
    # input_mask = f["masks/train/"+fl][...,1:3]#.reshape(-1,size_out,size_out,2)
    # x=unet.predict([test_nrcs,input_mask],verbose=1)

    # f.require_dataset("results/train/"+fl,shape=(nb_train,size_out,size_out,nbClass),dtype='f4',exact=False)
    # f["results/train/"+fl][:]=x.reshape(-1,size_out,size_out,nbClass) 
    # f.require_dataset("segmentation/train/"+fl,shape=(nb_train,size_out,size_out),dtype='i8',exact=False)
    # f["segmentation/train/"+fl][:]=m.to_classes(x).reshape(-1,size_out,size_out)
