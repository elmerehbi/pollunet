import numpy as np
import h5py as h
import matplotlib.pyplot as plt
from math import sqrt
from random import randint, shuffle
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix, accuracy_score

def to_classes(a):
    return np.argmax(a,axis=-1)

def stat(mask):
    return {i:(float(np.count_nonzero(mask==i)))/mask.size for i in np.unique(mask)}

def contains_mask(m,patch):
    return (patch & m).any()

def contains_class(c,patch):
    return (patch==c).any()

def frac_mask(m,a):
    c=0
    for i in a:
        c+=contains_mask(m,i)
    return 1.*c/len(a)

def pixelwise_error(s,t):
    a=to_classes(t)
    return 1-np.count_nonzero(s==a)/s.size

def iou(s,t,i):
    a=to_classes(t)
    return 1.*np.count_nonzero(s==i and a==i)/np.count_nonzero(s==i or a==i)

def nb_fx_neg(s,t,n):
    a=to_classes(t)
    c=0
    for i in range(len(s)):
        c+=contains_class(n,s[i]) and not contains_class(n,a[i])
    return c

def nb_fx_pos(s,t,n):
    a=to_classes(t)
    c=0
    for i in range(len(s)):
        c+= not contains_class(n,s[i]) and contains_class(n,a[i])
    return c

hdf="/users/local/h17valen/Deep_learning_pollution/data.hdf5"

def square(a):
    return a.reshape(int(sqrt(a.size)),-1)

def crop_as(a,b):
    s = a.shape[-2]
    r = b.shape[-2]
    o = (s-r)/2
    return a[...,o:-o,o:-o]

def aff(fn,i,p,t=None):
    with h.File(hdf,'r') as f:
        if t:
            plt.title(t)
        (a,b)=f["segmentation/{}/{}".format(p,fn)][i].shape
        o=(508-a)/2
        plt.subplot(1,3,1)
        if p=='train':
            plt.imshow(f["train/Nrcs/"][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        else:
            plt.imshow(f["test/Nrcs/"+p][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        plt.subplot(1,3,2)
        a,b,c=f["masks/{}/{}".format(p,fn)][i].shape
        d,e=f["segmentation/{}/{}".format(p,fn)][i].shape
        o=(a-d)/2
        plt.imshow(square(to_classes(f["masks/{}/{}".format(p,fn)][i][o:-o,o:-o]))) 
        plt.subplot(1,3,3)
        plt.imshow(f["segmentation/{}/{}".format(p,fn)][i])
        plt.show()

def aff_n(n,fn,i,p,t=None):
    with h.File(hdf,'r') as f:
        if t:
            plt.title(t)
        (a,b)=f["segmentation/{}/{}".format(p,fn)][i].shape
        o=(508-a)/2
        plt.subplot(1,3,1)
        if p=='train':
            plt.imshow(f["train/Nrcs/"][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        else:
            plt.imshow(f["test/Nrcs/"+p][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        plt.subplot(1,3,2)
        a,b,c=f["masks/{}/{}".format(p,fn)][i].shape
        d,e,g=f["results/{}/{}".format(p,fn)][i].shape
        o=(a-d)/2
        plt.imshow(square(to_classes(f["masks/{}/{}".format(p,fn)][i,o:-o,o:-o])))
        plt.subplot(1,3,3)
        plt.imshow(f["results/{}/{}".format(p,fn)][i,...,n])
        plt.show()

def view_rand(n,fn,p):
    f=h.File(hdf,'r')
    l=len(f['segmentation/{}/{}'.format(p,fn)])
    f.close()
    for i in range(n):
        m=randint(0,l-1)
        aff(fn,m,p,t=str(m))

def find_class(c,fn,p):
    l=[]
    with h.File(hdf,'r') as f:
        d=f["masks/{}/{}".format(p,fn)][:]
        d=to_classes(d)
        for i in range(len(d)):
            if contains_class(c,d[i]):
                l.append(i)
    return l

def find_mask(m,p):
    l=[]
    with h.File(hdf,'r') as f:
        if p == 'train':
            p='train/Mask'
        else:
            p='test/Mask/'+p
        d=f[p][:]
        for i in range(len(d)):
            if contains_mask(m,d[i]):
                l.append(i)
    return l

def find_detection(c,fn,p):
    l=[]
    with h.File(hdf,'r') as f:
        for i in range(len(f["segmentation/{}/{}".format(p,fn)])):
            if contains_class(c,f["segmentation/{}/{}".format(p,fn)][:]):
                l.append(i)
        return l

def view_list(l,fn,p):
    for i in l:
        aff(fn,i,p)

def view_n_list(n,l,fn,p):
    for i in l:
        aff_n(n,fn,i,p)

def view_class(c,fn,p):
    with h.File(hdf,'r') as f:
        d=f["masks/{}/{}".format(p,fn)][:]
        d=to_classes(d)
        for i in range(len(d)):
            if contains_class(c,d[i]):
                aff(fn,i,p)

def confusion(fn,p):
    with h.File(hdf,'r') as f:
        gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)]).reshape((-1,))
        pred=f["segmentation/{}/{}".format(p,fn)][:].reshape((-1,))
        return confusion_matrix(gt,pred)

def accuracy(fn,p):
    with h.File(hdf,'r') as f:
        gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)]).reshape(-1)
        pred=f["segmentation/{}/{}".format(p,fn)][:].reshape(-1)
        return accuracy_score(gt,pred,normalize=True)

def tx_fp(c,fn,p):
    with h.File(hdf,'r') as f:
        d=0
        l=len(f['masks/{}/{}'.format(p,fn)])
        n=0
        for i in range(l):
            a=to_classes(f['masks/{}/{}'.format(p,fn)][i,:])
            if not contains_class(c,a):
                n+=1
                if contains_class(c,f['segmentation/{}/{}'.format(p,fn)][i,:]):
                    d+=1
        if n != 0:
            return 1.*d/n
        else:
            return 0

def tx_fn(c,fn,p):
    with h.File(hdf,'r') as f:
        d=0
        n=0
        l=len(f['masks/{}/{}'.format(p,fn)])
        for i in range(l):
            if contains_class(c,to_classes(f['masks/{}/{}'.format(p,fn)][i,:])):
                n+=1
                if not contains_class(c,f['segmentation/{}/{}'.format(p,fn)][i,:]):
                    d+=1
        if n != 0:
            return 1.*d/n
        else:
            return 0 

def measures(fn,c=2):
    with h.File(hdf,'r') as f: 
        for p,pr in [('testing_images','\nTesting dataset\n'),('training_images','\nTesting dataset\n')]:
            print pr
            gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)][:]).reshape((-1,))
            pred=f["segmentation/{}/{}".format(p,fn)][:].reshape((-1,))
            print "confusion:"
            a=confusion_matrix(gt,pred)
            a/=np.sum(a,axis=-1)[:,None]
            print a
            print "accuracy:"
            print accuracy_score(gt,pred)
            del gt,pred
            print "false positive:"
            print tx_fp(c,fn,p)
            print "false negative:"
            print tx_fn(c,fn,p)


def randl(n,lg,l):
    return sorted(set([randint(0,lg) for i in range(n-len(l))]+l))

l4_train=[130, 160, 245, 262, 446, 459, 690, 849, 855, 1101, 1147, 1148, 1159, 1207, 1219, 1285, 1288, 1372, 1383, 1384, 1397, 1414, 1422, 1429, 1439, 1687, 2332, 2346, 2352, 2353, 2435, 2439, 2462, 2473, 2521, 2587, 2675, 2753, 3157, 3198, 3528, 3767, 3859, 3881, 3892, 4201, 4223, 4225, 4231, 4256, 4297, 4326, 4358, 4361, 4363, 4392, 4514, 4525, 4533, 4570, 4586, 5287, 5315, 5334, 5367, 5567, 6112, 6113, 6135, 6144, 6152, 6170, 6223, 6265, 6288, 6307, 6318, 6330, 6427, 6444, 6457, 6479, 6480, 6489, 6520, 6536, 6877, 6917, 7184, 7186, 7198, 7220, 7221, 7228, 7292, 7451, 7469, 7646, 7694, 7704, 7859, 7871, 7994, 8035, 8134, 8343, 8354, 8435, 8436, 8466, 8470, 8490, 8501, 8505, 8522, 8540, 8832, 8911, 8913, 8914, 8941, 9103, 9133, 9317, 9669, 9856, 9894, 9913, 10075, 10084, 10113, 10179, 10200, 10206, 10652, 10822, 10977, 11007, 11056, 11117, 11122, 11135, 11148, 11150, 11173, 11181, 11197, 11202, 11209, 11228, 11270, 11285, 11289, 11789, 12560, 13339, 13367, 13531, 13552, 13584, 13610, 14064, 14070, 14077, 14091, 14098, 14680, 14694, 14717, 14725, 15008, 15210, 15276, 15306, 15373, 15435, 15436, 15442, 15451, 15465, 15881, 15904, 15909, 15944, 16041, 16743, 16779, 16825, 17037]

l4_tr=[15, 123, 221, 355, 411, 590, 603, 718, 791, 807, 848, 876, 1179, 1410, 1520, 1573, 1623, 1628]

l4_ts=[90, 94, 108, 362, 377, 401, 411, 446, 487, 734, 739, 871, 876, 883, 887, 924, 1164, 1191, 1245, 1249, 1262, 1517, 1580, 1586, 1786, 1811, 1817, 1825, 1952, 1963, 1970, 1999, 2016, 2029, 2038, 2063, 2071, 2387, 2653, 2674, 3097, 3466, 3472, 3544, 3570, 3571, 3664, 3740]

tr='training_images'
ts='testing_images'
tra='train'
