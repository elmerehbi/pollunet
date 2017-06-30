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

# Load Segmentation maps training data (from 2000 to 2010)
#SegmaskTot=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/SegmaskTot_2000_2010.npy')
# load SSH AVISO maps data (2011 will be used for validation data)
#SSH_aviso_train=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/SSH_aviso_2000_2011.npy')

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

width = 108
height = 108
nbClass = 3

kernel = 5



img_input = Input(shape=(height, width))

######################################ENCODER

x = Reshape((1,height,width))(img_input)

x = Cropping2D(200,data_format="channels_first")(x)

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

conv3 = Cropping2D(4,data_format="channels_first")(conv3)
up1 = concatenate([UpSampling2D(size=(2, 2))(x), conv3], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up1)
x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)

conv2 = Cropping2D(16,data_format="channels_first")(conv2)
up2 = concatenate([UpSampling2D(size=(2, 2))(x), conv2], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up2)
x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)

conv1 = Cropping2D(40,data_format="channels_first")(conv1)
up3 = concatenate([UpSampling2D(size=(2, 2))(x), conv1], axis=1)
x = Conv2D(32, (kernel, kernel), padding="valid", kernel_initializer='he_normal',data_format="channels_first")(up3)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
x = Activation("elu")(x)  

####################################### Segmentation Layer

x = Conv2D(nbClass, (1, 1), padding="valid",data_format="channels_first")(x) 
# x = Reshape((nbClass, -1))(x) 

x = Permute((3, 2, 1))(x)
x = Activation("softmax")(x)
eddynet = Model(img_input, x)
############################################################################################# COMPILE

print eddynet.summary()

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
    
    # Train nrcs
    trn=f["train/Nrcs"]#[::]
    # Test nrcs
    tsn=f["test/Nrcs/testing_images"]

    if not f.__contains__("masks/training_images/"+fl):
        a=f["train/Mask"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        f["masks/train/"+fl].resize(a.shape+(3,))
        f.require_dataset("masks/train/"+fl,a.shape+(3,),maxshape=(None,420,420,3),dtype='f4',data=np_utils.to_categorical(a,3).reshape(a.shape+(3,)))
        f.create_dataset("masks/testing_images/"+fl,(0,420,420,3),maxshape=(None,420,420,3),dtype='f4')
        a=f["test/Mask/testing_images/"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        f["masks/testing_images/"+fl].resize(a.shape+(3,))
        f["masks/testing_images/"+fl][:]=np_utils.to_categorical(a,3).reshape(a.shape+(3,))
        a=f["test/Mask/training_images/"][:,44:464,44:464]
        a&=5
        a[a==4]=2
        a%=4
        f.create_dataset("masks/training_images/"+fl,(0,420,420,3),maxshape=(None,420,420,3),dtype='f4')
        f["masks/training_images/"+fl].resize(a.shape+(3,))
        f["masks/training_images/"+fl][:]=np_utils.to_categorical(a,3).reshape(a.shape+(3,))

    # Train maskn
    trm=f["masks/train/"+fl]#[::]
    # Test mask
    tsm=f["masks/testing_images/"+fl]

    nb_batch=10

    l=len(trn)/nb_batch

    for i in range(nb_batch):
        eddynet.fit(trn[i*l:(i+1)*l],trm[i*l:(i+1)*l],shuffle=True,verbose=2)
    
    v=eddynet.predict(tsn)

#
# from sklearn.feature_extraction.image import extract_patches_2d

# pheight=64
# pwidth=64
# max_patches=500
# random_state=1 #for reproductivity

# window_shape = (pheight, pwidth)
# nbdaysTrain=400
# strideDays=10
# nbpatchs=max_patches

# BB=np.zeros((nbdaysTrain*nbpatchs,pheight,pwidth))
# BB_label=np.zeros((nbdaysTrain*nbpatchs,pheight,pwidth)).astype(int)

# for dayN in range(nbdaysTrain):
#     BB[dayN*nbpatchs:dayN*nbpatchs+nbpatchs,:,:] = extract_patches_2d(SSH_aviso_train[:,:,strideDays*dayN+1], window_shape, max_patches, random_state)
#     BB_label[dayN*nbpatchs:dayN*nbpatchs+nbpatchs,:,:] = extract_patches_2d(SegmaskTot[:,:,strideDays*dayN], window_shape, max_patches, random_state)

# ### Validation dataset
# Segmask2011=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/segmask2011.npy')
# nbdaysTest=30
# strideDaysTest=10
# nbpatchsTest=1000

# BB_test=np.zeros((nbdaysTest*nbpatchsTest,pheight,pwidth))
# BB_test_label=np.zeros((nbdaysTest*nbpatchsTest,pheight,pwidth)).astype(int)
# for dayN in range(nbdaysTest):
#     BB_test[dayN*nbpatchsTest:dayN*nbpatchsTest+nbpatchsTest,:,:] = extract_patches_2d(SSH_aviso_train[:,:,strideDaysTest*dayN+4019], window_shape, nbpatchsTest, random_state)
#     BB_test_label[dayN*nbpatchsTest:dayN*nbpatchsTest+nbpatchsTest,:,:] = extract_patches_2d(Segmask2011[:,:,strideDaysTest*dayN], window_shape, nbpatchsTest, random_state)

# ###   
# x_train = np.reshape(BB, (len(BB), 1, pheight, pwidth))
# x_train[x_train<-100]=0

# label_train=np.reshape(BB_label, (len(BB), 1, pheight*pwidth)).transpose((0,2,1))
# x_train_label=np.zeros((len(BB),pheight*pwidth,nbClass))
# for kk in range(len(BB)):
#     print kk
#     x_train_label[kk,:,:] = np_utils.to_categorical(label_train[kk,:,:],nbClass)
    
 
# x_test = np.reshape(BB_test, (len(BB_test), 1, pheight, pwidth))
# x_test[x_test<-100]=0

# label_test=np.reshape(BB_test_label, (len(BB_test), 1, pheight*pwidth)).transpose((0,2,1))
# x_test_label=np.zeros((len(BB_test),pheight*pwidth,nbClass))
# for kk in range(len(BB_test)):
#     print kk
#     x_test_label[kk,:,:] = np_utils.to_categorical(label_test[kk,:,:],nbClass)

# ### cleaning  
# del label_test  
# del BB_test, SSH_aviso_train, BB_test_label 
# del SegmaskTot, Segmask2011

# ### Class weights     
# labels_dict={0:label_train.flatten().tolist().count(0),
#             1:label_train.flatten().tolist().count(1),
#             2:label_train.flatten().tolist().count(2),
#             3:label_train.flatten().tolist().count(3)}
            
# del label_train

# #from sklearn.utils import compute_class_weight
# #class_weight=compute_class_weight('balanced',np.unique(label_train),label_train.flatten())
# class_weight=[np.sum(labels_dict.values()) / float((nbClass * labels_dict[i])) for i in range(nbClass)]
# #class_weight={0:'1.',1:'5.',2:'5.',3:'0.1'}
# #class_weight=[0.35,5.25,8.68,1.14]
# #class_weight=[1.0,2.0,2.0,0.1]
# sample_weight=np.reshape(BB_label, (len(BB_label), pheight*pwidth)).copy().astype('float64')
# sample_weight[sample_weight==1]=class_weight[1]
# sample_weight[sample_weight==0]=class_weight[0]  #important to start with 1  before 0 class_weight[0] is equal to 1
# sample_weight[sample_weight==2]=class_weight[2]
# sample_weight[sample_weight==3]=class_weight[3]

# ### cleaning  
# del BB_label 
 
# ############################################### EDDYNET

                     
# eddyhist=eddynet.fit(x_train, x_train_label,
#                     epochs=2,
#                     batch_size=32,
#                     shuffle=True,
#                     validation_data=(x_test, x_test_label),
#                     sample_weight=sample_weight)  
#                     #class_weight = [0.,1000.,1000.,0.]) 
#                     #class_weight = 'auto')
                                   
# # eddynet.save('UnetmodelEddynet_kern5_weights1201_elu.h5')

# ## returns a compiled model
# ## identical to the previous one
# #eddynet1 = load_model('modelEddynet.h5')

# ########################################### AVISO TEST 2012

# SSH_aviso_2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/SSH_aviso_2012.npy')
# SSH_aviso_2012[SSH_aviso_2012<-100]=0

# Segmask2012=np.load('/homes/rlguensa/Bureau/Sanssauvegarde/segmaskyears/segmask2012.npy')

# ########################################### extract random 2012 SSH patches
# nbdaysfrom2012=30
# strideDays2012=10
# nbpatchs2012=100

# Test2012=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth))
# Test2012_label=np.zeros((nbdaysfrom2012*nbpatchs2012,pheight,pwidth)).astype(int)
# for dayN in range(nbdaysfrom2012):
#     Test2012[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(SSH_aviso_2012[:,:,strideDays2012*dayN+1], window_shape, nbpatchs2012, random_state)
#     Test2012_label[dayN*nbpatchs2012:dayN*nbpatchs2012+nbpatchs2012,:,:] = extract_patches_2d(Segmask2012[:,:,strideDays2012*dayN], window_shape, nbpatchs2012, random_state)


# ############################ Patch plot
# plt.figure()
# plt.ion()
# for i in range(10):
#     # display original
#     ax = plt.subplot(1, 3, 1)
#     plt.imshow(Test2012[i+20,:,:])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
    

#     # display ground truth segm
#     ax = plt.subplot(1, 3, 2)
#     plt.imshow(Test2012_label[i+20,:,:])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)


#     # display reconstruction
#     ax = plt.subplot(1, 3, 3)
#     predictSeg=eddynet.predict(np.reshape(Test2012[i+20,:,:],(1,1,height,width)))
#     plt.imshow(np.reshape(predictSeg.argmax(2).T,(pheight,pwidth)))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False) 
    
#     plt.show()
#     plt.draw()
#     #plt.pause(1)
#     plt.waitforbuttonpress()
    
# ################"" Confusion Matrix
# from sklearn.metrics import confusion_matrix

# print(confusion_matrix(Test2012_label[i+20,:,:].flatten(), np.argmax(predictSeg[0,:,:], axis=1)))
# #####################  HISTORY
# ## list all data in history
# print(eddyhist.history.keys())
# # summarize history for loss
# plt.plot(eddyhist.history['loss'])
# plt.plot(eddyhist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# plt.plot(eddyhist.history['acc'])
# #
# ################################# Segmentation of bigger maps

# widthAviso=280
# heightAviso=240
 
# from skimage.util.shape import view_as_windows
             
# patchesAviso=view_as_windows(SSH_aviso_2012[:,:,100], (32, 32),32)
# for h in range(len(patchesAviso)):
#     np.pad(patchesAviso, (2, 3), 'reflect')
