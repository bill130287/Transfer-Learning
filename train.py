# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:38:23 2019

@author: bill
"""

# -*- coding: utf-8 -*-

"""
Created on Sun Oct 13 2019
@author: PC-Chen

"""

import numpy as np
import cv2
import os
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image
#from keras.preprocessing.image import img_to_array

###############################################################################

now = datetime.datetime.now

###############################################################################

def img_mapping(loc):
    imgs=[]
    labels=[]
    classes=os.listdir(loc)
    for clas in classes:
        imlist=os.listdir(loc+clas)
        for im in imlist:
            imgs.append(loc+clas+'/'+im)
            labels.append(clas)
        print(len(imgs),'img_mapping'+loc+clas)
        
    return (imgs,labels)

###############################################################################

def minibatch_generator(imgs,labels,batchsize):

    images=[]    
    l=len(imgs)
    rand=np.random.uniform(0,l,batchsize).astype(np.int).tolist()
    img_batch=[imgs[i] for i in rand]
    labels_batch=[labels[i] for i in rand]

    for indx,img in enumerate(img_batch):

        i=cv2.imread(img).astype('float32')
        #i=Image.open(img)
        #i=np.array(i)
        i=i.reshape((1,3,i.shape[0],i.shape[1]))
        i=i/255
        images.append(i)       
    return (images,labels_batch)   

###############################################################################

def read_image_tranform_dimension(imgs,labels):

    images=[]    
    l=len(imgs)

    for indx,img in enumerate(imgs):

        i=cv2.imread(img).astype('float32')
        #i=Image.open(img)
        #i=np.array(i)
        i=i.reshape((1,3,i.shape[0],i.shape[1]))
        i=i/255
        images.append(i)
        print(len(images),'read_image_tranform_dimension',indx)
    return (images,labels)   

###############################################################################



def main():

    train_root= os.getcwd() + '/create_dataset/train/'
    (imgs,labels)=img_mapping(train_root)
    # labels_onehot = np_utils.to_categorical(labels)
    # np_utils.to_categorical(y_train, num_classes=10)

    val_root= os.getcwd() + '/create_dataset/val/'
    (val_imgs,val_labels)=img_mapping(val_root)    
    # print(len(val_imgs))
    (val_imgs,val_labels)=read_image_tranform_dimension(val_imgs,val_labels)
    # print(val_labels)
    # val_labels_onehot = np_utils.to_categorical(val_labels)
    
    tranfer_train_root= os.getcwd() + '/tranfer_dataset/train/'
    (tran_imgs,tran_labels)=img_mapping(tranfer_train_root)
    # print(tran_labels)
    # tran_labels_onehot = np_utils.to_categorical(tran_labels)

    tranfer_val_root= os.getcwd() + '/tranfer_dataset/val/'
    (tran_val_imgs,tran_val_labels)=img_mapping(tranfer_val_root)    
    (tran_val_imgs,tran_val_labels)=read_image_tranform_dimension(tran_val_imgs,tran_val_labels)
    # print(tran_val_labels)
    # tran_val_labels_onehot = np_utils.to_categorical(tran_val_labels)
    # print('--------------------' , tran_val_labels)

    batchsize=128
    epochs=2

#    datagen=augment(imgs,labels,1024)

    ## start training
    feature_layers = [
        Convolution2D(128, 3, 3, border_mode='same', input_shape=(3, 128, 128), activation='relu'),
        Conv2D(3,kernel_size=(2,2),activation='relu'),
        Conv2D(2,kernel_size=(2,2),activation='relu'),
        Conv2D(1,kernel_size=(2,2),activation='relu'),
        MaxPooling2D(pool_size=(5,5)),

        Dropout(0.2),
        Flatten(),
    ]
    classification_layers = [
        Dense(1024),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(32),
        Activation('relu'),

        Dense(10),
        Activation('softmax')
    ]    
    model = Sequential(feature_layers + classification_layers)
    print('---------------------------------------------' , classification_layers[3] , '---------------------------------------------')
    
    # Compile model
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
    
    mb_per_epoch=int(len(imgs)/batchsize)
    t = now()
    val_imgs = np.vstack(val_imgs)
    val_labels = np.array(val_labels)
    for epoch in range(epochs):
        print ('Entering epoch no ' , epoch)
        for mb_number in range(mb_per_epoch):
            print ('Entering minibatch no ',mb_number, 'out of ',mb_per_epoch)
            (mb_train,mb_labels)=minibatch_generator(imgs,labels,batchsize)
            mb_labels=np.array(mb_labels)
            
            model.fit(np.vstack(mb_train),
                        mb_labels, 
                        batch_size=batchsize, 
                        epochs=1, 
                        validation_data=(val_imgs,val_labels) )

        model.save('train_model.h5')

    print('-------------------------------', 'Original Training time: %s' % (now() - t), '-------------------------------')
    score = model.evaluate(val_imgs, val_labels, verbose=0)
    print('-------------------------------', 'score:', score[0], '-------------------------------')
    print('-------------------------------', 'accuracy:', score[1], '-------------------------------')
    print('-------------------------------', model.summary(), '-------------------------------')

    for l in feature_layers:
        l.trainable = False

    tran_mb_per_epoch=int(len(tran_imgs)/batchsize)
    t = now()
    tran_val_imgs = np.vstack(tran_val_imgs)
    tran_val_labels = np.array(tran_val_labels)
    for epoch in range(epochs):
        print ('Entering epoch no ' , epoch)
        for mb_number in range(tran_mb_per_epoch):
            print ('Entering minibatch no ',mb_number, 'out of ',tran_mb_per_epoch)
            (mb_train,mb_labels)=minibatch_generator(tran_imgs,tran_labels,batchsize)
            mb_labels=np.array(mb_labels)
            model.fit(np.vstack(mb_train),
                        mb_labels, 
                        batch_size=batchsize, 
                        epochs=1, 
                        validation_data=(tran_val_imgs,tran_val_labels) )

        model.save('tranfer_train_model.h5')

    print('-------------------------------', 'Original Training time: %s' % (now() - t), '-------------------------------')
    score = model.evaluate(tran_val_imgs, tran_val_labels, verbose=0)
    print('-------------------------------', 'score:', score[0], '-------------------------------')
    print('-------------------------------', 'accuracy:', score[1], '-------------------------------')
    print('-------------------------------', model.summary(), '-------------------------------')

if __name__=='__main__':

    main()