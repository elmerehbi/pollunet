#coding=utf8
"""Fonctions d'extraction des patchs"""

from netCDF4 import Dataset
from matplotlib.pyplot import show, imshow, subplot
from matplotlib.colors import LogNorm
from random import randint
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
#from copy import deepcopy
from numpy.lib.stride_tricks import as_strided
from os import listdir
import h5py as h

#########################################

# Taille des patches: (Il doivent pouvoir passer par U-net)
patch_size = 508
# Nombre max d'image 508x508 d'apprentissage à extraire des images d'apprentissage:
nb_max_patch_par_image = 200
# Fraction minimale de l'image qui doit être recouverte de mer:
min_sea_fraction = 0.60
# Bordure commune max entre une image 508x508 et les autres (en pixels):
max_overlap = 200
# Nombre de patch de test par image d'apprentissage:
nb_test_patch_par_image = 10
# Emplacement des fichiers
# Fichiers netcdf de départ:
netcdf_dir="/users/local/h17valen/Deep_learning_pollution/Database_netcdf/{}/{}"
# Fichier hdf extrait:
hdf = "/users/local/h17valen/Deep_learning_pollution/data.hdf5"

#########################################

def import_field(name,field='Nrcs'):
    fh=Dataset(name)
    a=fh.variables[field][:]
    fh.close()
    return a

def import_image(name):
    fh=Dataset(name,mode='r',format="NETCDF4")
#    a=deepcopy(fh.variables)
    a={key:value[:] for key,value in fh.variables.items()}
    fh.close()
    return a

def aff(name,field='Nrcs'):
    a=import_field(name,field)
    if field == 'Nrcs':
#        a[a>0.05]=0.05
#        norm=None
        norm=LogNorm()
    else:
        norm=None
    imshow(a[::10,::10],norm=norm,origin='lc')
    show()

def rand_patch(f,size=patch_size,min_sea_fraction=0.75):
    a,r=f['Nrcs'].shape
    x,y=randint(0,a-size),randint(0,r-size)
    while np.count_nonzero(f['Mask'][x:x+size,y:y+size]!=1)<=min_sea_fraction*size**2:
        x,y=randint(0,a-size),randint(0,r-size)
    return x,y

def tile_array(a, b1, b2):
    l, r, c = a.shape
    ls, rs, cs = a.strides
    x = as_strided(a, (l, r, b1, c, b2), (ls, rs, 0, cs, 0))
    return x.reshape(l, r*b1, c*b2)

###############################################################

coef = np.zeros(26)

# init CMOD-IFR2 coef

coef[0]  =  0.0
coef[1]  = -2.437597
coef[2]  = -1.5670307
coef[3]  =  0.3708242
coef[4]  = -0.040590
coef[5]  =  0.404678
coef[6]  =  0.188397
coef[7]  = -0.027262
coef[8]  =  0.064650
coef[9]  =  0.054500
coef[10] =  0.086350
coef[11] =  0.055100
coef[12] =  -0.058450
coef[13] = -0.096100
coef[14] = 0.412754
coef[15] =  0.121785
coef[16] =  -0.024333
coef[17] = 0.072163
coef[18] =  -0.062954
coef[19] =  0.015958
coef[20] = -0.069514
coef[21] = -0.062945
coef[22] =  0.035538
coef[23] = 0.023049
coef[24] =  0.074654
coef[25] =  -0.014713



def funcB1(x, V):
    C=coef
    
    tetamin = 18.
    tetamax = 58.
    tetanor = (2.*x - (tetamin+tetamax))/(tetamax-tetamin)
    vmin = 3.
    vmax = 25.
    vitnor = (2.*V - (vmax+vmin))/(vmax-vmin)
    pv0 = 1.
    pv1 = vitnor
    pv2 = 2*vitnor*pv1 - pv0
    pv3 = 2*vitnor*pv2 - pv1
    pt0 = 1.
    pt1 = tetanor
    pt2 = 2*tetanor*pt1 - pt0
    pt3 = 2*tetanor*pt2 - pt1
    result = C[8] + C[9]*pv1 + (C[10]+C[11]*pv1)*pt1 + (C[12]+C[13]*pv1)*pt2
    return result

def funcB2(x, V):
    C=coef
    
    tetamin = 18.
    tetamax = 58.
    tetanor = (2.*x - (tetamin+tetamax))/(tetamax-tetamin)
    vmin = 3.
    vmax = 25.
    vitnor = (2.*V - (vmax+vmin))/(vmax-vmin)
    pv0 = 1.
    pv1 = vitnor
    pv2 = 2*vitnor*pv1 - pv0
    pv3 = 2*vitnor*pv2 - pv1
    pt0 = 1.
    pt1 = tetanor
    pt2 = 2*tetanor*pt1 - pt0
    pt3 = 2*tetanor*pt2 - pt1
    result = C[14] + C[15]*pt1 + C[16]*pt2 + \
             (C[17]+C[18]*pt1+C[19]*pt2)*pv1 + (C[20]+C[21]*pt1+C[22]*pt2)*pv2 + \
             (C[23]+C[24]*pt1+C[25]*pt2)*pv3
    return result

def _getNRCS(inc_angle, wind_speed, wind_dir):
    """
    ; Renvoie une NRCS prédite par modèle IFR2
    ;
    ; INPUTS:
    ;  inc_angle:
    ;  wind_speed:
    ;  wind_dir:
    ;
    ; OPTIONAL INPUTS:
    ;  wind_speed:
    ;
    """
    
    T=inc_angle
    wind=wind_speed
    
    C = coef
    
    tetai = (T - 36.)/19.
    xSQ = tetai*tetai
    P0 = 1.
    P1 = tetai
    P2 = (3.*xSQ-1.)/2.
    P3 = (5.*xSQ-3.)*tetai/2.
    ALPH = C[1] + C[2] * P1 + C[3] * P2 + C[4]*P3
    BETA = C[5] + C[6] * P1 + C[7] * P2
    ang = wind_dir
    cosi  = np.cos(np.deg2rad(ang))
    cos2i = 2.*cosi*cosi - 1.
    b1   = funcB1(T, wind)
    b2   = funcB2(T, wind)
    b0 = np.power(10., (ALPH+BETA*np.sqrt(wind)))
    sig  = b0 * ( 1. + b1*cosi + np.tanh(b2)*cos2i )
    return sig

def view_gmf(fn):
    nrcs=import_field(fn,'Nrcs')[::10,::10]
    ws=import_field(fn,'modelWindSpeed')
    wd=import_field(fn,'modelWindDirection')
    ia=import_field(fn,'Incidence angle')
    ws=tile_array(ws[None,...],25,25)[0,::10,0::10][:nrcs.shape[0],:nrcs.shape[1]]
    wd=tile_array(wd[None,...],25,25)[0,::10,0::10][:nrcs.shape[0],:nrcs.shape[1]]
    ia=ia[::10][:ws.shape[1]]
#    print ia.shape,nrcs.shape,wd.shape,ws.shape
    gmf=_getNRCS(ia,ws,wd)
    vmax=gmf.max()
    vmin=min(nrcs.min(),gmf.min())
    nrcs[nrcs>2*vmax]=2*vmax
    subplot(1,4,1)
    imshow(nrcs,cmap='gray',vmin=vmin,vmax=2*vmax)
    subplot(1,4,2)
    imshow(gmf,cmap='gray',norm=LogNorm())#,vmin=vmin,vmax=vmax)
    subplot(1,4,3)
    imshow(wd,norm=LogNorm())
    subplot(1,4,4)
    imshow(ws,norm=LogNorm()) 
    show()

###################################################################



def extract_pts(f,n=nb_max_patch_par_image,size=patch_size,min_sea_fraction=min_sea_fraction,max_overlap=max_overlap):
    mask=np.zeros(f['Nrcs'].shape)
    l=[]

    a,r=f['Nrcs'].shape
    for i in range(n):
        c=0
        x,y=rand_patch(f,size=size,min_sea_fraction=min_sea_fraction)
        while np.count_nonzero(mask[x+max_overlap:x+size-max_overlap,y+max_overlap:y+size-max_overlap]==1)>0:
            c+=1
            if c>100:
                break
            x,y=rand_patch(f,size=size,min_sea_fraction=min_sea_fraction)
        if c>100:
            break
        mask[x:x+size,y:y+size]=1
        l.append((x,y))
    return l

def extract_patches(f,pts,size,c=1):
    return [f[int(x/c):int(x/c)+int(size),int(y/c):int(y/c)+int(size)] for x,y in pts]

def calculate_patches(hdf=hdf,nc=netcdf_dir):
    files_train=listdir(netcdf_dir.format("train",""))
    files_test=listdir(netcdf_dir.format("test",""))
    with h.File(hdf,'w') as f:
        for fn in files_test:
            fh=import_image(netcdf_dir.format("test",fn))
            pts=extract_pts(fh)
            if f.__contains__("test/patches/{}".format(fn)):
                f.__delitem__("test/patches/{}".format(fn))
            f.create_dataset("test/patches/{}".format(fn),data=np.array(pts))
        for fn in files_train:
            fh=import_image(netcdf_dir.format("train",fn))
            pts=extract_pts(fh)
            if f.__contains__("test/patches/{}".format(fn)):
                f.__delitem__("test/patches/{}".format(fn))
            f.create_dataset("test/patches/{}".format(fn),data=np.array(pts[:10]))
            if f.__contains__("train/patches/{}".format(fn)):
                f.__delitem__("train/patches/{}".format(fn))
            f.create_dataset("train/patches/{}".format(fn),data=np.array(pts[10:]))
            
def load_testing_images(field,size,hdf=hdf,nc=netcdf_dir,dt=None,c=1,broadcast=False):
    files_test=listdir(netcdf_dir.format("test",""))
    with h.File(hdf,"a") as f:
        if f.__contains__("test/{}/testing_images".format(field)):
            f.__delitem__("test/{}/testing_images".format(field))
        ttest=f.create_dataset("test/{}/testing_images".format(field),shape=(0,size,size),chunks=True,maxshape=(None,size,size),dtype=dt)
        for file_name in files_test:
            fh=import_field(netcdf_dir.format("test",file_name),field=field)
            if broadcast:
                b=import_field(netcdf_dir.format("test",file_name),field='Nrcs')
                fh=np.broadcast_to(fh,b.shape)

            pts=f["test/patches"][file_name]
            n=ttest.shape[0]
            ttest.resize(n+len(pts),0)
            for j,p in enumerate(extract_patches(fh,pts,size,c=c)):
                if p.shape[0]<int(size):
                    m=int(size)-p.shape[-2]
                    p=np.concatenate((p,)+(m*(p[[-1]],)),axis=0)
                if p.shape[1]<int(size):
                    m=int(size)-p.shape[-1]
                    p=np.concatenate((p,)+(m*(p[:,[-1]],)),axis=1)
                ttest[n+j]=p
        
def load_training_images(field,size,hdf=hdf,nc=netcdf_dir,dt=None,c=1,broadcast=False):
    files_train=listdir(netcdf_dir.format("train","")) 
    with h.File(hdf,"a") as f:
        if f.__contains__("test/{}/training_images".format(field)):
            f.__delitem__("test/{}/training_images".format(field))
        test=f.create_dataset("test/{}/training_images".format(field),(0,size,size),chunks=True,maxshape=(None,size,size),dtype=dt)
        if f.__contains__("train/{}".format(field)):
            f.__delitem__("train/{}".format(field))
        dataset_size=sum([len(i) for i in f['train/patches']])
        train=f.create_dataset("train/{}".format(field),(dataset_size,size,size),chunks=True,dtype=dt)
        m,n=0,0
        for fn in files_train:
            fh=import_field(netcdf_dir.format("train",fn),field=field)
            if broadcast:
                b=import_field(netcdf_dir.format("train",fn),field='Nrcs')
                fh=np.broadcast_to(fh,b.shape)
            pts_train=f["train/patches"][fn]
            pts_test=f["test/patches"][fn]
            n=test.shape[0]
            m=train.shape[0]
            test.resize(n+len(pts_test),0)
            train.resize(m+len(pts_train),0)
            for j,p in enumerate(extract_patches(fh,pts_test,size,c=c)):
                if p.shape[0]<int(size):
                    mq=int(size)-p.shape[-2]
                    p=np.concatenate((p,)+(mq*(p[[-1]],)),axis=0)
                if p.shape[1]<int(size):
                    mq=int(size)-p.shape[-1]
                    p=np.concatenate((p,)+(mq*(p[:,[-1]],)),axis=1)
                test[n+j]=p
            for j,p in enumerate(extract_patches(fh,pts_train,size,c=c)):
                if p.shape[0]<int(size):
                    mq=int(size)-p.shape[-2]
                    p=np.concatenate((p,)+(mq*(p[[-1]],)),axis=0)
                if p.shape[1]<int(size):
                    mq=int(size)-p.shape[-1]
                    p=np.concatenate((p,)+(mq*(p[:,[-1]],)),axis=1)
                train[m+j]=p

def extract_angle(size):
    field = 'Incidence angle'
    dt='float32'
    with h.File(hdf,"a") as f:
        if f.__contains__("train/{}".format(field)):
            f.__delitem__("train/{}".format(field))
        train=f.create_dataset("train/{}".format(field),(0,size,size),chunks=True,maxshape=(None,size,size),dtype=dt)
        if f.__contains__("test/{}/training_images".format(field)):
            f.__delitem__("test/{}/training_images".format(field))
        test=f.create_dataset("test/{}/training_images".format(field),(0,size,size),chunks=True,maxshape=(None,size,size),dtype=dt)
        for fn in listdir(netcdf_dir.format("train","")):
            a=import_field(netcdf_dir.format("train",fn),field=field)
            print a.shape
            print fn
            pts_train = f["train/patches"][fn]
            pts_test = f["test/patches"][fn]
            patches_train = [a[i:i+size] for j,i in pts_train]
            patches_test = [a[i:i+size] for j,i in pts_test]
            n=test.shape[0]
            m=train.shape[0]
            test.resize(n+len(pts_test),0)
            train.resize(m+len(pts_train),0)
            for j,p in enumerate(patches_train):
#                print train.shape, train[n+j].shape, p.shape
                train[n+j]=p
            for j,p in enumerate(patches_test):
                test[m+j]=p
        if f.__contains__("test/{}/testing_images".format(field)):
            f.__delitem__("test/{}/testing_images".format(field))
            test=f.create_dataset("test/{}/training_images".format(field),(0,size,size),chunks=True,maxshape=(None,size,size),dtype=dt)
        for fn in listdir(netcdf_dir.format("test","")):
            a=import_field(netcdf_dir.format("train",fn),field=field)
            pts = f["test/patches"][fn]
            patches = [a[i,i+size] for j,i in pts]
            n=test.shape[0]
            test.resize(m+len(pts_train),0) 
            for j,p in patches:
                train[n+j]=p

def load_field(field,size,hdf=hdf,nc=netcdf_dir,c=1,broadcast=False):
    if field=='Mask':
        dt='int8'
    else:
        dt='float32'
    load_testing_images(field,size,hdf=hdf,nc=netcdf_dir,dt=dt,c=c,broadcast=broadcast)
    load_training_images(field,size,hdf=hdf,nc=netcdf_dir,dt=dt,c=c,broadcast=broadcast)
def change_mask(name,hdf=hdf):
    with h.File(hdf,"a") as f:
        u=f["train/Mask"]
        v=f["test/Mask/testing_images"]
        w=f["test/Mask/training_images"]

        
########################################
# Scripts d'extraction, à décommenter
########################################


# print "calculating patches..."
# calculate_patches()
# print "Ok"
# print "extracting masks..."
# load_field('Mask',patch_size)
# print "Ok"
# print "extracting nrcs images..."
# load_field('Nrcs',patch_size)
# print "Ok"


# print "extracting wind direction"

# load_field('modelWindDirection',patch_size/25.+1.,c=25.)

# print "extracting wind speed"

# load_field('modelWindSpeed',patch_size/25.+1.,c=25.) 

# print "extracting incidence angle"
# load_field('Incidence angle',patch_size,broadcast=True)
# print "Ok"

# print 'calculating gmf'

# with h.File(hdf) as f:
#     n=12
#     for p in ['train/{}','test/{}/testing_images','test/{}/training_images']:
#         if f.__contains__(p.format('GMF')):
#             f.__delitem__(p.format('GMF'))
#         f.create_dataset(p.format('GMF'),f[p.format('Nrcs')].shape,dtype='float32')
#         for i in range(n):
#             l=len(f[p.format('modelWindSpeed')])/n+1
#             ws=tile_array(f[p.format('modelWindSpeed')][i*l:(i+1)*l],25,25)[:,0:patch_size,0:patch_size]
#             wd=tile_array(f[p.format('modelWindDirection')][i*l:(i+1)*l],25,25)[:,0:patch_size,0:patch_size]
#             ia=f[p.format('Incidence angle')][i*l:(i+1)*l]
#             print ws.shape, wd.shape, ia.shape
#             f[p.format('GMF')][i*l:(i+1)*l]=_getNRCS(ia, ws, wd)



def unchunk(field,dataset):
    if dataset == 'train':
        path='train/'+field
    else:
        path='test/{}/{}'.format(field,dataset)
    with h.File(hdf) as f:
        f.create_dataset('tmp',f[path].shape,data=f[path])
        del f[path]
        f[path]=f['tmp']
        del f['tmp']

# print 'n'
# unchunk('Nrcs')
# print 'm'
# unchunk('Mask')
# print 'wd'
# unchunk('modelWindDirection')
# print 'ws'
# unchunk('modelWindSpeed')

for f in ('Mask','Nrcs'):
    for d in ('testing_images','training_images'):
        print f,d
        unchunk(f,d)
