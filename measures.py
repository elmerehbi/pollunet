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
            a=a/np.sum(a,axis=-1)[:,None].astype(float)
            print a
            print confusion(fn,p)
            print "accuracy:"
            print accuracy_score(gt,pred)
            del gt,pred
            print "false positive:"
            print tx_fp(c,fn,p)
            print "false negative:"
            print tx_fn(c,fn,p)


def randl(n,lg,l):
    return sorted(set([randint(0,lg) for i in range(n-len(l))]+l))

tr='training_images'
ts='testing_images'
tra='train'

def load_pollutions():
    with h.File(hdf,'a') as f:
        if f.__contains__('pollutions_indices/'+tra):
            del f['pollutions_indices/'+tra]
        if f.__contains__('pollutions_indices/'+tr):
            del f['pollutions_indices/'+tr]
        if f.__contains__('pollutions_indices/'+ts):
            del f['pollutions_indices/'+ts]
        f['pollutions_indices/'+tra]=np.array(find_mask(6,tra))
        f['pollutions_indices/'+tr]=np.array(find_mask(6,tr))
        f['pollutions_indices/'+ts]=np.array(find_mask(6,ts))

#load_pollutions()

with h.File(hdf,'r') as f: 
    l4_train=list(f['pollutions_indices/'+tra][:])
    l4_tr=list(f['pollutions_indices/'+tr][:])
    l4_ts=list(f['pollutions_indices/'+ts][:])
