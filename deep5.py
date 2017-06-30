

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
from keras import optimizers
import h5py as h
import sys
sys.path.append('../')
import measures as m
import os.path 

###########################

K.set_image_dim_ordering('th') # Theano dimension ordering in this code

nb_train = 10629
nb_set1 = 1700
nb_set2 = 2471

# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/h17valen/Deep_learning_pollution/weights/"

# Network parameters:
width = 508
height = 508
nbClass = 3
kernel = 5
depth = 4
nb_conv = 1
nb_conv_out = 1
activation = "relu"
dropout_down = 0.9
fdropout_up = 0.9

channels_max = 64

def channels(depth):
    return channels_max

# Training parameters
batch_size=16
epochs=4
optimizer=optimizers.SGD(lr=0.1)

###########################

def crop():
    c=kernel-1
    for i in range(depth):
        yield c
        c+=(kernel-1)*nb_conv
        c*=2

print list(crop())

size_out = width - 2 * list(crop())[-1] - (kernel-1) * (nb_conv_out + nb_conv)

c=crop()


def unet_layers(x,depth):
    if depth == 0:
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
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
        x=Conv2D(channels(depth),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
    return x

###########################

img_input = Input(shape=(height, width))

x = Reshape((1,height,width))(img_input)

x = unet_layers(x,depth)

for i in range(nb_conv_out):
    x = Conv2D(nbClass,1,padding='valid',data_format='channels_first')(x)

x = Reshape((nbClass,-1))(x)

x = Permute((2,1))(x)

x = Activation('softmax')(x)

unet = Model(img_input,x)

###########################

# print unet.summary()
# exit()
# print size_out

unet.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'],
                sample_weight_mode="temporal")


###########################

hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"
fl='deep5.py'#__file__
print fl

with h.File(hdf,"a") as f:
  
    if not f.__contains__("weights/"+fl):
        b=size_out*size_out
        o=(width-size_out)/2
        a=f["train/Mask"][:,o:-o,o:-o][:]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
        f.require_dataset("masks/train/"+fl,a.shape+(nbClass,),dtype='f4')
        n=2
        l=len(a)/n
        for i in range(n):
            f["masks/train/"+fl][i*l:(i+1)*l]=np_utils.to_categorical(a[i*l:(i+1)*l],nbClass).reshape(a[i*l:(i+1)*l].shape+(nbClass,))
        f.require_dataset("masks/testing_images/"+fl,(nb_set2,size_out*size_out,nbClass),dtype='f4')
        a=f["test/Mask/testing_images/"][:,o:-o,o:-o][:]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
#        f["masks/testing_images/"+fl].resize(a.shape+(nbClass,))
        f["masks/testing_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))
        a=f["test/Mask/training_images/"][:,o:-o,o:-o][:]
        a&=5
        a[a==4]=2
        a%=4
        a=a.reshape(-1,b)
        f.require_dataset("masks/training_images/"+fl,(nb_set1,size_out*size_out,nbClass),dtype='f4')
#        f["masks/training_images/"+fl].resize(a.shape+(nbClass,))
        f["masks/training_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))

        (a,b,c)=f["masks/train/"+fl].shape
        w=np.full((a,b),1.,dtype=np.float32)
        q=np.argmax(f["masks/train/"+fl][:],axis=-1)
        w[q==1]=7.
        w[q==2]=10000.
        w=w.reshape(a,-1)
        del q
        f.require_dataset('weights/'+fl,(nb_train,b),dtype=np.float32,exact=False)
        f["weights/"+fl][:]=w


###########################

    
    # Weights
    w=f["weights/"+fl]
    # Train nrcs
    train_nrcs=f["train/Nrcs"]#[::]
    # Test nrcs
    test_nrcs=f["test/Nrcs/testing_images"]#[:]

    # Train mask
    train_mask=f["masks/train/"+fl]#[::]
    
    # Test mask
    test_mask=f["masks/testing_images/"+fl]#[:]
    

    if os.path.isfile(weight_dir+fl):
        unet.load_weights(weight_dir+fl)

    w=w[m.l4_train]
    train_nrcs=train_nrcs[m.l4_train]
    train_mask=train_mask[m.l4_train]
    batch_size=2
    epochs=24

    unet.fit(train_nrcs,train_mask,shuffle='batch',verbose=1,batch_size=batch_size,epochs=epochs,sample_weight=w)

    unet.save_weights(weight_dir+fl)

#    print unet.evaluate(test_nrcs,test_mask)

    # v=unet.predict(test_nrcs,verbose=1)

    # f.require_dataset("results/testing_images/"+fl,shape=(nb_set2,size_out,size_out,nbClass),dtype='f4',exact=False)
    # f["results/testing_images/"+fl][:]=v.reshape(-1,size_out,size_out,nbClass)
    # f.require_dataset("segmentation/testing_images/"+fl,shape=(nb_set2,size_out,size_out),dtype='i8',exact=False)
    # f["segmentation/testing_images/"+fl][:]=m.to_classes(v).reshape(-1,size_out,size_out)

    # test_nrcs=f["test/Nrcs/training_images/"]
    # w=unet.predict(test_nrcs,verbose=1)

    # f.require_dataset("results/training_images/"+fl,shape=(nb_set1,size_out,size_out,nbClass),dtype='f4',exact=False)
    # f["results/training_images/"+fl][:]=w.reshape(-1,size_out,size_out,nbClass) 
    # f.require_dataset("segmentation/training_images/"+fl,shape=(nb_set1,size_out,size_out),dtype='i8',exact=False)
    # f["segmentation/training_images/"+fl][:]=m.to_classes(w).reshape(-1,size_out,size_out)

    # test_nrcs=f["train/Nrcs/"]
    # x=unet.predict(test_nrcs,verbose=1)

    # f.require_dataset("results/train/"+fl,shape=(nb_train,size_out,size_out,nbClass),dtype='f4',exact=False)
    # f["results/train/"+fl][:]=x.reshape(-1,size_out,size_out,nbClass) 
    # f.require_dataset("segmentation/train/"+fl,shape=(nb_train,size_out,size_out),dtype='i8',exact=False)
    # f["segmentation/train/"+fl][:]=m.to_classes(x).reshape(-1,size_out,size_out)
