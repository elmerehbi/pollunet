# coding=utf8

print __file__

import numpy as np
import h5py as h
from keras.utils import np_utils

hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"

with h.File(hdf,"r") as f:
    
    # Train nrcs
    trn=f["train/Nrcs"]
    # Train mask
    trm=f["train/Mask"]
    # Test nrcs
    tsn=f["test/Nrcs/testing_images"]
    # Test mask
    tsm=f["test/Mask/testing_images"]
    
    # On commence avec deux classes: les pollutions (OSN et OSW, finies et en cours) et le reste
    # Le masque de départ a comme valeurs possibles: 0,1,8,12,18,20,32, 10,12,18 et 20 étant les pollutions
    trm=trm[:] & 4
    trm >>= 2
#    print np.unique(trm)
#    print np.bincount(trm)
    # print np.count_nonzero(trm == 12)
    # print np.count_nonzero(trm == 1)
    # print np.count_nonzero(trm == 0)
    # print trm.size
    # print trm.shape
    # trm=((trm & 4) >> 2)
    print trm.shape
    (a,b,c)=trm.shape
    a=np.zeros((a,b*c,2),dtype=np.float32)
    print a.size, a.shape
    for i in range(len(a)):
        a[i,:]=np_utils.to_categorical(trm[i],2)
    print a
    

#    tsm=np_utils.to_categorical(tsm,2).reshape(a,b,c,2)

    # eddynet.fit(trn,trm,shuffle=True)

    # v=eddynet.predict(tsn)
