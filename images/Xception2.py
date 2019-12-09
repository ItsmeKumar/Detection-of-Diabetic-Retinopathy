#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import cv2
#from resnets_utils import *

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint


# In[ ]:


import time
import tensorflow as tf
from sklearn import metrics
import pickle as pi
from collections import Counter


# In[5]:


import gc
gc.collect()


# In[ ]:


# import pickle
# f=open("/content/drive/My Drive/DS5500_Project_2/processed_images/merged_array","rb")
# try:
#     while True:
#         x=pickle.load(f)
#         # print x
# except EOFError:
#     pass
# f.close()


# In[ ]:


# with open('/content/drive/My Drive/DS5500_Project_2/processed_images/merged_array', mode='rb') as file: # b is important -> binary
#     merged_array = file.read()
# np.shape(merged_array)


# In[ ]:


images_labels = np.load('/content/drive/My Drive/DS5500_Project_2/processed_images/prathwish_test_data/images_labels_2.npy',allow_pickle=True)


# In[ ]:


class_0 = [[x[0],x[1]] for x in images_labels if x[1]==0][:500].copy()
class_1 = [[x[0],x[1]] for x in images_labels if x[1]==1][:500].copy()
class_2 = [[x[0],x[1]] for x in images_labels if x[1]==2][:500].copy()
class_3 = [[x[0],x[1]] for x in images_labels if x[1]==3][:500].copy()
class_4 = [[x[0],x[1]] for x in images_labels if x[1]==4][:500].copy()


# In[ ]:


images_labels2 = np.concatenate((class_0,class_1,class_2,class_3,class_4),axis=0)


# In[11]:


del images_labels,class_0,class_1,class_2,class_3,class_4
gc.collect()


# In[ ]:


def subtract_gaussian_blur(img,b=5):
    gb_img = cv2.GaussianBlur(img, (0, 0), b)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)


# In[ ]:


train_images = [subtract_gaussian_blur(x[0],35) for x in images_labels2]
train_labels = [x[1] for x in images_labels2]


# In[14]:


len(train_images)


# In[15]:


import seaborn as sns
#Lets plot the label to be sure we just have two class
sns.countplot(train_labels)
plt.title('Distribution')


# In[16]:


plt.figure(figsize=[20,5])
for row in range(6):
    img = train_images[row]
    plt.subplot(1,6,row+1)
    plt.imshow(img)


# In[17]:


del images_labels2
gc.collect()


# In[18]:


X = np.array(train_images)
y = np.array(train_labels)
from keras.utils import to_categorical
y = to_categorical(y)
del train_images,train_labels
gc.collect()
print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[ ]:


#Lets split the data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32

#This helps prevent overfitting, since we are using a small dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale

#Create the image generators
train_generator = train_datagen.flow(X_train, y_train,batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# In[ ]:


from keras.applications.xception import Xception


# In[21]:


img_sz = 512
img_height,img_width = img_sz, img_sz
num_classes = 5
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = Xception(weights= None, include_top=False, input_shape= (img_height,img_width,3))


# In[22]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation= 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam, Adadelta
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adadelta(lr=0.01)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[23]:


from keras.optimizers import SGD, Adam, Adadelta
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adadelta(lr=0.01)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[24]:


history= model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)


# In[ ]:


print('start time:'+str(time.time()))
model.fit(X_train, train_labels  , epochs = 20, batch_size = 128)
print('end time:'+str(time.time()))


# In[ ]:


preds = model.evaluate(X_test, test_labels)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:


pred_value=model.predict(X_train)
val_labels= train_labels.copy()


# In[ ]:


# np.shape(val_labels)
pred_value


# In[ ]:



pred=np.argmax(pred_value,axis=1)
actual=np.argmax(val_labels,axis=1)
accuracy= metrics.accuracy_score(actual,pred)
# precision= metrics.average_precision_score(actual,pred)
# f1_score= metrics.f1_score(actual,pred)
print("accuracy: "+str(accuracy))
# print("precision:"+ str(precision))
# print("f1 score: "+ str(f1_score))
print("kappa Score: "+str(metrics.cohen_kappa_score(actual,pred, weights='quadratic')))
print("confusion matrix")
print(metrics.confusion_matrix(actual, pred))


# In[ ]:


print(metrics.classification_report(actual, pred, digits=3))


# In[ ]:



print(Counter(actual))
print(Counter(pred))


# In[ ]:


calculate_metrics(pred_value,val_labels)


# In[ ]:


metrics.confusion_matrix(actual, pred)


# In[ ]:


type(pred_value)


# In[ ]:


def calculate_metrics(pred, actual):
  pred=np.argmax(pred)
  actual=np.argmax(actual)
  accuracy= metrics.accuracy_score(actual,pred)
  precision= metrics.average_precision_score(actual,pred)
  f1_score= metrics.f1_score(actual,pred)
  print("accuracy: "+str(accuracy))
  print("precision:"+ str(precision))
  print("f1 score: "+ str(f1_score))
  print("confusion matrix")
  print(metrics.confusion_matrix(actual, pred))


# In[ ]:


calculate_metrics(pred_value,val_set)


# In[ ]:


np.argmax(pred_value,axis=1)


# In[ ]:


np.argmax(val_labels, axis=1)


# In[ ]:





# In[ ]:




