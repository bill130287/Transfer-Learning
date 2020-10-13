import numpy as np
import cv2
import os
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from keras.models import load_model

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
    return (imgs,labels)

###############################################################################

def read_image_tranform_dimension(imgs,labels):

    images=[]    
    l=len(imgs)

    for indx,img in enumerate(imgs):

        i=cv2.imread(img).astype('float32')
        i=i.reshape((1,3,i.shape[0],i.shape[1]))
        i=i/255
        images.append(i)
    return (images,labels)   


###############################################################################
def plot_image(images, labels,predict):
	
	fig = plt.gcf()
	plt.imshow(np.reshape(images, (128,128,3)), cmap='binary')
	plt.title("Label= "+labels+',\n'+"Predict= "+predict)

###############################################################################

model = load_model('tranfer_train_model_test.h5')
predict_root= os.getcwd() + '/prediction/'
(pred_imgs,pred_labels)=img_mapping(predict_root) 
print(pred_imgs)
(pred_imgs,pred_labels)=read_image_tranform_dimension(pred_imgs,pred_labels)
results=[]
plt.figure(figsize=(20,10))

for i in range(10):
  #i=test_imgs[???,:]
  print('Shape : ', pred_imgs[i].shape) 
  #i=i.reshape((1,3,pred_imgs.shape[0],pred_imgs.shape[1]))
  results.append(model.predict_classes(pred_imgs[i]))
  plt.subplot(2,5,i+1)
  plot_image(pred_imgs[i],chr(int(pred_labels[i])+97),chr(int(results[i])+97))
  #print('number: ',results[i])
  #print('Result : ', chr(int(results[i])+97) )
plt.tight_layout()
plt.savefig('Prediction.png',dpi=300,format='png')
print("Prediction.png is saved")
plt.show()
