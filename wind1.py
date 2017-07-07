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
from img_extract import load_field

###########################
# Paramètres du modèle
###########################

# Dossier de sauvegarde des poids des modèles
weight_dir = "/users/local/h17valen/Deep_learning_pollution/weights/"

# Poids des différentes classes pour le loss
######
weight_pollution = 5000.
weight_land = 5.
weight_boats = 1000.
weight_sea = 1.
reload_weights = False

# Paramètres du réseau
######

# Taille des images en entrée
width = 508
height = 508
# Nombre de classes à distinguer
nbClass = 3
# Taille du kernel utilisé pour les convolutions
kernel = 5
# Profondeur du réseau U-net (en nombre de max-pooling/upsampling)
depth = 4
# Nombre de convolutions après chaque max-pooling/upsampling
nb_conv = 1
# Nombre de convolutions en sortie
nb_conv_out = 2
# Activation
activation = "relu"
# Dropout dans la première et la deuxième moitié du réseau (résolution descendant et montante)
dropout_down = 0.1
dropout_up = 0.1

# Nombre max de channels utilisés dans le réseau
channels_max = 48

# Fonction qui donne le nombre de channels à utiliser en fonction de la profondeur (profondeur décroissante: elle vaut 0 en bas du réseau)
def channels(depth, channels_max=channels_max):
    return channels_max
#    return channels_max / 2**depth

# Apprentissage
######
# Nombre de patch utilisés
training_size = 2000
# Batch size (Préferer un grand batch_size, mais si il est trop grand le réseau ne rentre plus en mémoire)
batch_size=12
# Nombre d'époques sur un choix d'images
epochs=2
# Nombre de fois qu'on choisit training_size images
nb_shuffle=0
# Optimizer
optimizer=adadelta()


###########################
# Génération du réseau
###########################


K.set_image_dim_ordering('th') # Theano dimension ordering in this code

# Nombre de pixels à rogner sur les images
def crop():
    c=kernel-1
    for i in range(depth):
        yield c
        c+=(kernel-1)*nb_conv
        c*=2


size_out = width - 2 * list(crop())[-1] - (kernel-1) * (nb_conv_out + 2 * nb_conv)

c=crop()


def unet_layers(x,depth,channels_max):
    """
    Construit récursivement les couches U-net à partir de la couche x.

    Inputs: 
        x: keras layer
        depth: profondeur du réseau
        channels_max: 
    """
    if depth == 0:
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
        x=Conv2D(channels(depth,channels_max)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_down)(x)
    else:
        for i in range(nb_conv):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_down)(x)
    if depth>0:
        y=MaxPooling2D(pool_size=(2,2))(x)
        y=unet_layers(y,depth-1,channels_max)
        y=UpSampling2D(size=(2,2))(y)
        x=Cropping2D(c.next(),data_format='channels_first')(x)
        x=concatenate([x,y],axis=1)
        for i in range(nb_conv-1):
            x=Conv2D(channels(depth,channels_max),kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
            x=Dropout(dropout_up)(x)
        x=Conv2D(channels(depth,channels_max)/2,kernel,padding='valid',activation=activation,kernel_initializer='he_normal',data_format='channels_first')(x)
        x=Dropout(dropout_up)(x)
    return x

###########################
# Construction du réseau
###########################

sar_input = Input(shape=(height, width))
gmf_input = Input(shape=(height, width))
mask_input = Input(shape=(height,width))

sar = Reshape((1,height,width))(sar_input)
gmf = Reshape((1,height,width))(gmf_input) 
mask = Reshape((1,height,width))(mask_input) #Permute((3,1,2))(mask_input)

x = concatenate([sar,gmf],axis=1)

x = unet_layers(x,depth,channels_max)

c = crop()

y = unet_layers(mask,depth,8)

x = concatenate([y,x],axis=1)

for i in range(nb_conv_out):
    x = Conv2D(channels(depth),kernel,padding='valid',data_format='channels_first')(x)

x = Conv2D(nbClass,1,padding='valid',data_format='channels_first')(x)

x = Reshape((nbClass,-1))(x)

x = Permute((2,1))(x)

x = Activation('softmax')(x)

unet = Model([sar_input,gmf_input,mask_input],x)

# print unet.summary()
# exit()
# print size_out
unet.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'],
                sample_weight_mode="temporal")


###########################
# Chargement des masques
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

    if not f.__contains__("masks/training_images/"+fl):
        a='gmf3.py'
        f['masks/train/'+fl]=f['masks/train/'+a]
        f['masks/testing_images/'+fl]=f['masks/testing_images/'+a]
        f['masks/training_images/'+fl]=f['masks/training_images/'+a]
        f['weights/'+fl]=f['weights/'+a]

    
    # Nombre d'étape pour traiter 
    n=8
    if not f.__contains__("masks/training_images/"+fl):
        a=f["train/Mask"][:]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=1
        f.require_dataset("masks/train/"+fl,a.shape+(nbClass,),dtype='f4')

        l=len(a)/n+1
        for i in range(n):
            f["masks/train/"+fl][i*l:(i+1)*l]=np_utils.to_categorical(a[i*l:(i+1)*l],nbClass).reshape(a[i*l:(i+1)*l].shape+(nbClass,))
        a=f["test/Mask/testing_images/"][:]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=1
        f.require_dataset("masks/testing_images/"+fl,a.shape+(nbClass,),dtype='f4')
        f["masks/testing_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))
        a=f["test/Mask/training_images/"][:]
        a[(a&6)!=0]=2
        a[(a&32)!=0]=1
        f.require_dataset("masks/training_images/"+fl,a.shape+(nbClass,),dtype='f4')
        f["masks/training_images/"+fl][:]=np_utils.to_categorical(a,nbClass).reshape(a.shape+(nbClass,))
        
    if not f.__contains__("weights/"+fl) or reload_weights:
        (a,b,d,c)=f["masks/train/"+fl].shape
        w=np.full((a,size_out*size_out),1.,dtype=np.float32)
        l=len(w)/n+1
        for i in range(n):
            b=f["train/Mask/"][i*l:(i+1)*l,o:-o,o:-o].reshape((-1,size_out*size_out))
            b[(a&6)]=2
            b[(a&32)]=1
            w[i*l:(i+1)*l,:]=b[:]
        w=w.reshape(a,-1)
        f.require_dataset('weights/'+fl,(nb_train,size_out*size_out),dtype=np.float32,exact=False)
        f["weights/"+fl][:]=w

    if not f.__contains__("test/modelWindSpeed/testing_images/"):
        load_field('modelWindSpeed',width,c=25.)

##########################
# Selection des patchs ayant un vent moyen entre 4 et 8 m/s
##########################

with h.File(hdf,"r") as f:
    avg=np.average(f['train/modelWindSpeed'][:],axis=(-1,-2))
    l_train=list(np.arange(nb_train)[np.logical_and(avg>4.,avg<8.)])
    print len(l_train)
    avg=np.average(f['test/modelWindSpeed/training_images'][:],axis=(-1,-2))
    l_tr=list(np.arange(nb_set1)[np.logical_and(avg>4.,avg<8.)])
    print len(l_tr)
    avg=np.average(f['test/modelWindSpeed/testing_images'][:],axis=(-1,-2))
    l_ts=list(np.arange(nb_set2)[np.logical_and(avg>4.,avg<8.)])
    print len(l_ts)

###########################
# Apprentissage
###########################

if os.path.isfile(weight_dir+fl):
    unet.load_weights(weight_dir+fl)


with h.File(hdf,"r") as f:
    print nb_shuffle
    for i in range(nb_shuffle):
        print i
        l = l_train #m.randl(training_size,nb_train,m.l4_train)

        #    print len(l), nb_train,len(l), f["weights/"+fl].shape
        # print l[-1], nb_train
        # Weights
        print "weights"
        w=f["weights/"+fl][l]
        # Train nrcs
        print "nrcs"
        train_nrcs=f["train/Nrcs"][l]
        # Train mask
        print "mask"
        train_mask=f["masks/train/"+fl][l][:,o:-o,o:-o,:].reshape((-1,size_out*size_out,nbClass))
        # Input gmf
        print "gmf"
        input_gmf = f["train/GMF"][l]
        # Input mask
        print "mask"
        input_mask = f["masks/train/"+fl][l][...,1]
        print "weights"
        w[w==1]=weight_land
        w[w==0]=weight_sea   
        w[w==2]=weight_pollution
        w[w==3]=weight_boats
        
        print "fit"
        unet.fit([train_nrcs,input_gmf,input_mask],train_mask,shuffle=True,verbose=1,batch_size=batch_size,epochs=epochs,sample_weight=w)

        print "save"
        unet.save_weights(weight_dir+fl)

        del input_gmf
        del train_mask
        del train_nrcs

###########################
# Test sur les jeux de test
###########################

    l=l_ts
    test_nrcs=f["test/Nrcs/testing_images/"][l]
    input_gmf = f["test/GMF/testing_images"][l]
    input_mask = f["masks/testing_images/"+fl][l][...,1] 
    v=unet.predict([test_nrcs,input_gmf,input_mask],verbose=1,batch_size=16)

with h.File(hdf,"a") as f:
    f.require_dataset("results/testing_images/"+fl,shape=(len(test_nrcs),size_out,size_out,nbClass),dtype='f4',exact=False)
    f["results/testing_images/"+fl][:]=v.reshape(-1,size_out,size_out,nbClass)
    f.require_dataset("segmentation/testing_images/"+fl,shape=(len(test_nrcs),size_out,size_out),dtype='i8',exact=False)
    f["segmentation/testing_images/"+fl][:]=m.to_classes(v).reshape(-1,size_out,size_out)

    del input_gmf
    del test_nrcs

l=l_tr
with h.File(hdf,"r") as f:
    test_nrcs=f["test/Nrcs/training_images/"][l]
    input_gmf=f["test/GMF/training_images"][l]
    input_mask = f["masks/training_images/"+fl][l][...,1]#.reshape(-1,size_out,size_out,2)
    w=unet.predict([test_nrcs,input_gmf,input_mask],verbose=1,batch_size=16)

with h.File(hdf,"a") as f:
    f.require_dataset("results/training_images/"+fl,shape=(len(test_nrcs),size_out,size_out,nbClass),dtype='f4',exact=False)
    f["results/training_images/"+fl][:]=w.reshape(-1,size_out,size_out,nbClass) 
    f.require_dataset("segmentation/training_images/"+fl,shape=(len(test_nrcs),size_out,size_out),dtype='i8',exact=False)
    f["segmentation/training_images/"+fl][:]=m.to_classes(w).reshape(-1,size_out,size_out)

    del input_gmf
    del test_nrcs

    # test_nrcs=f["train/Nrcs/"]
    # input_mask = f["masks/train/"+fl][...,1:3]#.reshape(-1,size_out,size_out,2)
    # x=unet.predict([test_nrcs,input_mask],verbose=1)

    # f.require_dataset("results/train/"+fl,shape=(nb_train,size_out,size_out,nbClass),dtype='f4',exact=False)
    # f["results/train/"+fl][:]=x.reshape(-1,size_out,size_out,nbClass) 
    # f.require_dataset("segmentation/train/"+fl,shape=(nb_train,size_out,size_out),dtype='i8',exact=False)
    # f["segmentation/train/"+fl][:]=m.to_classes(x).reshape(-1,size_out,size_out)
