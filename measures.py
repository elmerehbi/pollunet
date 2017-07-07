#coding=utf8

"""
Fichier contenant des fonction diverses de mesure et de visualisation des résultats
"""

import numpy as np
import h5py as h
import matplotlib.pyplot as plt
from math import sqrt
from random import randint, shuffle
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix, accuracy_score
import numexpr as ne

def to_classes(a):
    """Renvoie les classes prédites à partir des probabilités"""
    return np.argmax(a,axis=-1)

def stat(mask):
    """Renvoie un dictionnaire contenant les fréquences des différentes valeurs d'un array"""
    return {i:(float(np.count_nonzero(mask==i)))/mask.size for i in np.unique(mask)}

def contains_mask(m,patch):
    """Teste si un patch contient le masque m (bitwise)"""
    return (patch & m).any()

def contains_class(c,patch):
    """Teste si un patch contient la classe c"""
    return (patch==c).any()

def frac_mask(m,a):
    """Calcule la fraction des patches dans un array a qui contiennent le masque m"""
    c=0
    for i in a:
        c+=contains_mask(m,i)
    return 1.*c/len(a)

def pixelwise_error(s,t):
    """Fraction de mauvaise prédiction"""
    a=to_classes(t)
    return 1-float(np.count_nonzero(s==a))/s.size

def iou(s,t,i):
    """Intersection des prediction over union"""
    a=to_classes(t)
    return 1.*np.count_nonzero(s==i and a==i)/np.count_nonzero(s==i or a==i)

def nb_fx_neg(s,t,n):
    """Nombre des patchs de s qur lesquels t ne détecte pas de pollution alors qu'il y en a"""
    a=to_classes(t)
    c=0
    for i in range(len(s)):
        c+=contains_class(n,s[i]) and not contains_class(n,a[i])
    return c

def nb_fx_pos(s,t,n):
    """Nombre des patchs de s qur lesquels t détecte de la pollution alors qu'il n'y en a pas"""
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
    """
    Affiche la segmentation par le réseau du fichier python fn du iè patch du jeu de donnée p
    Input:
    fn: nom du fichier python se terminant par .py
    i: indice du patch 
    p: jeu de donnée, parmi 'train', 'training_images' ou 'testing_images' (selon ce qui a été effectivement segmenté dans le fichier)
    t: Titre à donner à l'image
    """
    with h.File(hdf,'r') as f:
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
        plt.imshow(square(to_classes(f["masks/{}/{}".format(p,fn)][i][o:-o,o:-o])),vmin=0) 
        plt.subplot(1,3,3)
        plt.imshow(f["segmentation/{}/{}".format(p,fn)][i],vmin=0)
        if t:
            plt.suptitle(str(t))
        plt.show()

def aff_n(fn,i,p,n=2,t=None):
    """
    Affiche la probabilité prédite de la classe n par le réseau du fichier python fn du iè patch du jeu de donnée p
    Input:
    fn: nom du fichier python se terminant par .py
    i: indice du patch 
    p: jeu de donnée, parmi 'train', 'training_images' ou 'testing_images' (selon ce qui a été effectivement segmenté dans le fichier)
    n: numéro de la classe, la deuxième classe correspondant dans mes fichiers à la pollution
    t: titre à donner à l'image
    """
    with h.File(hdf,'r') as f:
        (a,b)=f["segmentation/{}/{}".format(p,fn)][i].shape
        o=(508-a)/2
        plt.subplot(2,2,1)
        if p=='train':
            plt.imshow(f["train/Nrcs/"][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        else:
            plt.imshow(f["test/Nrcs/"+p][i][o:-o,o:-o],norm=LogNorm(),cmap='gray')
        plt.subplot(2,2,2)
        a,b,c=f["masks/{}/{}".format(p,fn)][i].shape
        d,e,g=f["results/{}/{}".format(p,fn)][i].shape
        o=(a-d)/2
        plt.imshow(square(to_classes(f["masks/{}/{}".format(p,fn)][i,o:-o,o:-o])),vmin=0)
        plt.subplot(2,2,3)
        plt.imshow(f["results/{}/{}".format(p,fn)][i,...,n],vmin=0,vmax=1)
        plt.subplot(2,2,4)
#        print f["segmentation/{}/{}".format(p,fn)][i].shape
        plt.imshow(f["segmentation/{}/{}".format(p,fn)][i],vmin=0)
        if t:
            plt.suptitle(str(t))
        plt.show()

def aff_full(fn,i,p,n=2,t=None):
    """
    Affiche le résultat de la segmentation par le réseau du fichier python fn du iè patch du jeu de donnée p
    Input:
    fn: nom du fichier python se terminant par .py
    i: indice du patch 
    p: jeu de donnée, parmi 'train', 'training_images' ou 'testing_images' (selon ce qui a été effectivement segmenté dans le fichier)
    n: numéro de la classe, la deuxième classe correspondant dans mes fichiers à la pollution
    t: titre à donner à l'image
    """
    with h.File(hdf,'r') as f:
        (a,b)=f["segmentation/{}/{}".format(p,fn)][i].shape
        o=(508-a)/2
        q=o
        if p=='train':
            path='{1}/{0}'
        else:
            path='test/{0}/{1}'
        if f.__contains__(path.format("GMF",p)):
            gmf=f[path.format("GMF",p)][i][q:-q,q:-q]
            col=3
        else:
            col=2
        nrcs=f[path.format('Nrcs',p)][i][o:-o,o:-o]
        a,b,c=f["masks/{}/{}".format(p,fn)][i].shape
        d,e,g=f["results/{}/{}".format(p,fn)][i].shape
        o=(a-d)/2
        mask=square(to_classes(f["masks/{}/{}".format(p,fn)][i,o:-o,o:-o]))
        segm=f["segmentation/{}/{}".format(p,fn)][i]
        res=f["results/{}/{}".format(p,fn)][i,...,n]
        vmin=min(nrcs.min(),gmf.min())
        vmax=max(nrcs.max(),gmf.max())
        wmin=0
        wmax=3
        plt.subplot(2,col,1)
        plt.imshow(nrcs,norm=LogNorm(),cmap='gray',vmin=vmin,vmax=vmax)
        plt.title('Nrcs')
        plt.subplot(2,col,2)
        plt.imshow(mask,vmin=wmin,vmax=wmax)
        plt.title('Ground truth')
        if f.__contains__(path.format("GMF",p)):
            plt.subplot(2,3,3)
            plt.imshow(gmf,norm=LogNorm(),cmap='gray',vmin=vmin,vmax=vmax)
            plt.title('Geophysical model')
        plt.subplot(2,2,3)
        plt.imshow(segm,vmin=wmin,vmax=wmax)
        plt.title('Predicted segmentation')
        plt.subplot(2,2,4)
        plt.imshow(res,vmin=0,vmax=1)
        plt.title('Pollution probability')
        if t:
            plt.suptitle(str(t))
        plt.show()

def view_rand(n,fn,p):
    """Affiche le résultat sur n images aléatoires du réseau du fichier fn sur le jeu de données p"""
    f=h.File(hdf,'r')
    l=len(f['segmentation/{}/{}'.format(p,fn)])
    f.close()
    for i in range(n):
        m=randint(0,l-1)
        aff(fn,m,p,t=str(m))

def view_full_rand(n,fn,p):
    with h.File(hdf,'r') as f:
        l=len(f['segmentation/{}/{}'.format(p,fn)])
    for i in range(n):
        m=randint(0,l-1)
        aff_full(fn,m,p,t=str(m))

def find_class(c,fn,p):
    """Renvoie la liste des indices des masques en sortie contentant la classe c"""
    l=[]
    with h.File(hdf,'r') as f:
        d=f["masks/{}/{}".format(p,fn)][:]
        d=to_classes(d)
        for i in range(len(d)):
            if contains_class(c,d[i]):
                l.append(i)
    return l

def find_mask(m,p):
    """Renvoie la liste des indices des masques en sortie contentant le masque m dans le jeu de données p"""
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
    """Renvoie la liste des indices des patches pour lesquels le réseau du fichier fn a détecté la classe c"""
    l=[]
    with h.File(hdf,'r') as f:
        for i in range(len(f["segmentation/{}/{}".format(p,fn)])):
            if contains_class(c,f["segmentation/{}/{}".format(p,fn)][:]):
                l.append(i)
        return l

def view_list(l,fn,p):
    """Affiche la liste l avec la fonction aff"""
    for i in l:
        aff(fn,i,p)

def view_n_list(l,fn,p,n=2):
    """Affiche la liste l avec la fonction aff_n"""
    for i in l:
        aff_n(fn,i,p,n,i)

def view_full_list(l,fn,p,n=2):
    """Affiche la liste l avec la fonction aff_full"""
    for i in l:
        aff_full(fn,i,p,n,i)

def view_class(c,fn,p):
    """Affiche les résultats des patchs contenant la classe c"""
    with h.File(hdf,'r') as f:
        d=f["masks/{}/{}".format(p,fn)][:]
        d=to_classes(d)
        for i in range(len(d)):
            if contains_class(c,d[i]):
                aff(fn,i,p)

def confusion(fn,p):
    """Calcule la matrice de confusion pour la segmentation du jeu p par le réseau fn"""
    with h.File(hdf,'r') as f:
        gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)]).reshape((-1,))
        pred=f["segmentation/{}/{}".format(p,fn)][:].reshape((-1,))
        return confusion_matrix(gt,pred)

def accuracy(fn,p):
    """Calcule la précision pour la segmentation du jeu p par le réseau fn"""
    with h.File(hdf,'r') as f:
        gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)]).reshape(-1)
        pred=f["segmentation/{}/{}".format(p,fn)][:].reshape(-1)
        return accuracy_score(gt,pred,normalize=True)

def tx_fp(c,fn,p):
    """Fraction des patchs de s pour lesquels t détecte de la pollution alors qu'il n'y en a pas"""
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
    """Fraction des patchs de s pour lesquels t ne détecte pas de pollution alors qu'il y en a"""
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

def confusion_matrix(gt,pred):
    assert gt.shape == pred.shape
    n=gt.max()
    c=np.zeros((n+1,n+1),dtype='i8')
    for i in range(n):
        for j in range(n):
            c[i,j]=np.count_nonzero(ne.evaluate('(gt==i) & (pred==j)'))
    return c

def accuracy_score(gt,pred):
    assert gt.shape == pred.shape
    s=gt.size
    eq=np.count_nonzero(ne.evaluate('gt==pred'))
    return float(eq)/s

def fneg(gt,pred):
    assert gt.shape == pred.shape
    gt=np.any(ne.evaluate('gt == 2'),axis=(-1,-2))
    pred=np.any(ne.evaluate('pred == 2'),axis=(-1,-2))
    fp=np.count_nonzero(np.logical_and(pred,np.logical_not(gt)))
    fn=np.count_nonzero(np.logical_and(gt,np.logical_not(pred)))
    l=np.count_nonzero(gt)
    fp=float(fp)/l
    fn=float(fn)/(len(gt)-l)
    return fp, fn

def measures(fn,c=2):
    """Affiches les mesures choisies pour le réseau du fichier fn"""
    with h.File(hdf,'r') as f: 
        for p,pr in [('testing_images','\nTesting dataset\n'),('training_images','\nTesting dataset\n')]:
            print pr
            gt=crop_as(to_classes(f["masks/{}/{}".format(p,fn)]),f["segmentation/{}/{}".format(p,fn)][::20]).reshape((-1,))
            pred=f["segmentation/{}/{}".format(p,fn)][:].reshape((-1,))
            print "confusion:"
            a=confusion_matrix(gt,pred)
            a=a/np.sum(a,axis=-1)[:,None].astype(float)
            print a
            print "accuracy:"
            print accuracy_score(gt,pred)
            # del gt,pred
            print "false positive:"
            fp, fn = fneg(gt,pred)
            print fp
            # print tx_fp(c,fn,p)
            print "false negative:"
            print fn
            # print tx_fn(c,fn,p)


def randl(n,lg,l):
    """Renvoie une liste triée de longueur au plus n contenant la liste l et complétée avec des éléments aléatoires entre 0 et lg-1"""
    return sorted(set([randint(0,lg-1) for i in range(n-len(l))]+l))

tr='training_images'
ts='testing_images'
tra='train'

def load_pollutions():
    """"""
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
    if f.__contains__('pollutions_indices/'+ts):
        l4_train=list(f['pollutions_indices/'+tra][:])
        l4_tr=list(f['pollutions_indices/'+tr][:])
        l4_ts=list(f['pollutions_indices/'+ts][:])


# fppx=[0.0011,0.0012,0,0.0088,0.012,0.0075,0.0082]
# vppx=[0.2,0.15,0,0.15,0.53,0.516,0.504]
# fppt=[0.293,0.366,0,0.37,0.52,0.503,0.442,1]
# fnpt=[0.231,0.231,1,0.667,0.167,0.185,0.333,0]

# vppt=[1-i for i in fnpt]

# f=['base_weighted2','base_weighted3','deep1','deep6','land1','gmf1','gmf3','dummy']

# plt.plot(fppt,vppt,'o')

# for l,x,y in zip(f,fppt,vppt):
#     plt.annotate(l,xy=(x,y))

# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.title('Patchwise')

# plt.show()

# plt.plot(fppx,vppx,'o')

# for l,x,y in zip(f,fppx,vppx):
#     plt.annotate(l,xy=(x,y))

# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.title('Pixelwise')

# plt.show()


# fppx=[0.0019,0.0022,0.,0.0133,2.17284604e-02,1.37518478e-02,1.40178733e-02]
# vppx=[0.15025907,0.28196331,0.,0.01111681,7.98716747e-01,8.58842189e-01,8.06308598e-01]
# fppt=[0.363313609467,0.365680473373,0.0,0.407560543414,0.571342925659,0.575539568345,0.442351013721,1]
# fnpt=[0.1,0.0,1.0,0.714285714286,0.0625,0.03125,0.28125,0]

# vppt=[1-i for i in fnpt]

# f=['base_weighted2','base_weighted3','deep1','deep6','land1','gmf1','gmf3','dummy']

# plt.plot(fppt,vppt,'o')

# for l,x,y in zip(f,fppt,vppt):
#     plt.annotate(l,xy=(x,y))

# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.title('Patchwise')

# plt.show()

# plt.plot(fppx,vppx,'o')

# for l,x,y in zip(f,fppx,vppx):
#     plt.annotate(l,xy=(x,y))

# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.title('Pixelwise')

# plt.show()
