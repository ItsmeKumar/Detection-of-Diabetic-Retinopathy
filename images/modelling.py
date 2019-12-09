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


# In[ ]:


file = open('/content/drive/My Drive/DS5500_Project_2/processed_images/merged_array', 'rb')
merged_array = pi.load(file)
file.close()


# In[ ]:


merged_labels = pd.read_csv('/content/drive/My Drive/DS5500_Project_2/processed_images/all_labels.csv')


# In[ ]:


# Undersample each category to the size of count of least category

sample_size = merged_labels['diagnosis'].value_counts().min()
resampled_index = []
for category in np.sort(merged_labels['diagnosis'].unique()):
    temp_index = merged_labels[merged_labels['diagnosis']==category].sample(n=sample_size, replace=False, random_state=500).index
    resampled_index += list(temp_index)

merged_labels = merged_labels.loc[resampled_index,:]
merged_array = merged_array[resampled_index]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(merged_array, merged_labels['diagnosis'], test_size=0.2, random_state=500)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=500)


# In[ ]:


# df_train = pd.read_csv('/content/drive/My Drive/DS5500_Project_2/train.csv')
# df_test = pd.read_csv('/content/drive/My Drive/DS5500_Project_2/test.csv')

# x = df_train['id_code']
# y = df_train['diagnosis']

# x, y = shuffle(x, y, random_state=500)

# train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,
#                                                       stratify=y, random_state=500)


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


train_labels = to_categorical(train_y, 5)
val_labels = to_categorical(valid_y, 5)


# In[ ]:


# img_sz = 128

# def load_input(labels, IMG_SIZE=img_sz):
#   data_set = np.empty((labels.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
#   for i, id_code in enumerate(labels):
#     location=f"/content/drive/My Drive/DS5500_Project_2/train_images/{id_code}.png"
#     image = cv2.imread(location)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     data_set[i, :, :, :] = image
#   return data_set


# In[ ]:


# train_set = load_input(labels = train_x, IMG_SIZE=128)
# val_set = load_input(labels = valid_x, IMG_SIZE=128)
# train_labels = train_labels
# val_labels = val_labels


# In[ ]:


# file = open('/content/drive/My Drive/DS5500_Project_2/processed_images/train_set_col_res', 'rb')
# train_set = pi.load(file)
# file.close()

# file = open('/content/drive/My Drive/DS5500_Project_2/processed_images/val_set_col_res', 'rb')
# val_set = pi.load(file)
# file.close()


# In[ ]:


# train_size = 50
# val_size = 10
# train_set = load_input(labels = train_x[0:train_size], IMG_SIZE=128)
# val_set = load_input(labels = valid_x[0:val_size], IMG_SIZE=128)
# train_labels = train_labels[0:train_size]
# val_labels = val_labels[0:val_size]


# In[ ]:


from keras.applications.xception import Xception


# In[ ]:


img_height,img_width = img_sz, img_sz
num_classes = 5
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))


# In[ ]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation= 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)


# In[ ]:


from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(train_set, train_labels  , epochs = 30, batch_size = 64)


# In[ ]:


preds = model.evaluate(val_set, val_labels)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import os
os.chdir('/content/drive/My Drive/DS5500_Project_2/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


from glob import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
path_list = glob('./train_images/.png')[0:16]
path_list = './train_images/ffc04fed30e6.png'


# In[ ]:


path_list


# In[ ]:


def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2
    return (ry, rx)

def subtract_gaussian_blur(img,b=5):
    gb_img = cv2.GaussianBlur(img, (0, 0), b)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
    return a * b + 128 * (1 - b)

def crop_img(img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0
        crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]
        return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img
    return new_img
from skimage.color import rgb2gray,rgba2rgb
def preprocess(f, r, debug_plot=False):
    img = cv2.imread(f)
    ry, rx = estimate_radius(img)
    resize_scale = r / max(rx, ry)
    w = min(int(rx * resize_scale * 2), r * 2)
    h = min(int(ry * resize_scale * 2), r * 2)
    img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
    img = crop_img(img, h, w)
    if debug_plot:
        plt.figure()
        plt.imshow(img)
    img = subtract_gaussian_blur(img)
    img = remove_outer_circle(img, 0.9, r)
    img_rgba = np.zeros([img.shape[0],img.shape[1],4])
    for row in range(4):
        img2 = subtract_gaussian_blur(img,(row+1)*5)
        img_rgba[:,:,row] = rgb2gray(img2)
    img = place_in_square(img_rgba, r, h, w)
    if debug_plot:
        plt.figure()
        plt.imshow(img)
    return img_rgba


# In[ ]:


size= 512
r=size//2
path0 = './train_images/ffc04fed30e6.png'
# img= spaceboy
img = cv2.imread(path0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
#plt.imshow(img)


# In[ ]:


ry, rx = estimate_radius(img)
resize_scale = r / max(rx, ry)
w = min(int(rx * resize_scale * 2), r*2)
h = min(int(ry * resize_scale * 2), r*2)
img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
img = crop_img(img, h, w)
plt.xticks([])
plt.yticks([])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


img2 = subtract_gaussian_blur(img,50)
plt.xticks([])
plt.yticks([])
plt.imshow(img2)


# In[ ]:


img = remove_outer_circle(img2, 0.9, r)
plt.imshow(img)


# In[ ]:


plt.figure(figsize=[20,5])
for row in range(6):
    img2 = subtract_gaussian_blur(img,(row+1)*5)
    plt.subplot(1,6,row+1)
    plt.imshow(img2)


# In[ ]:


img2 = subtract_gaussian_blur(img,(6+1)*5)
plt.subplot(1,6,6+1)
plt.imshow(img2)


# In[ ]:


#make rgba data for training
from skimage.color import rgb2gray,rgba2rgb
plt.figure(figsize=[10,10])
img_rgba = np.zeros([img.shape[0],img.shape[1],4])
for row in range(4):
    img2 = subtract_gaussian_blur(img,(row+1)*5)
    img_rgba[:,:,row] = rgb2gray(img2)


# In[ ]:


plt.figure(figsize=[10,10])
img = crop_image_from_gray(img_rgba)
img = remove_outer_circle(img_rgba, 0.9, r)
plt.imshow(img)


# In[ ]:


# https://www.kaggle.com/orkatz2/diabetic-retinopathy-preprocess-rgba-update


# In[ ]:


import os
import glob
import cv2
import numpy as np

def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img)
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

spaceboy = circle_crop('./train_images/1df0a4c23c95.png')
ax[0].imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

cropboy = circle_crop('./train_images/0a1076183736.png')
ax[1].imshow(cv2.cvtColor(cropboy, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

squareboy = circle_crop('./train_images/0e3572b5884a.png')
ax[2].imshow(cv2.cvtColor(squareboy, cv2.COLOR_BGR2RGB))
ax[2].axis('off')

supercropboy = circle_crop('./train_images/698d6e422a80.png')
ax[3].imshow(cv2.cvtColor(supercropboy, cv2.COLOR_BGR2RGB))
ax[3].axis('off')


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

spaceboy = circle_crop_v2('./train_images/1df0a4c23c95.png')
ax[0].imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

cropboy = circle_crop_v2('./train_images/0a1076183736.png')
ax[1].imshow(cv2.cvtColor(cropboy, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

squareboy = circle_crop_v2('./train_images/0e3572b5884a.png')
ax[2].imshow(cv2.cvtColor(squareboy, cv2.COLOR_BGR2RGB))
ax[2].axis('off')

supercropboy = circle_crop_v2('./train_images/698d6e422a80.png')
ax[3].imshow(cv2.cvtColor(supercropboy, cv2.COLOR_BGR2RGB))
ax[3].axis('off')


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('./train_images/b69c224edd6e.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


spaceboy = circle_crop('./train_images/b69c224edd6e.png')
plt.imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))


# In[ ]:


path_list[0]


# In[ ]:


from PIL import Image               # to load images
from IPython.display import display 
im = Image.open('./train_images/b69c224edd6e.png')


# In[ ]:


imshow(im)


# In[ ]:


path_list[0]


# In[ ]:


from PIL import Image               # to load images
from IPython.display import display 
im = Image.open('./train_images/ffec9a18a3ce.png')
plt.imshow(im)


# In[ ]:


spaceboy = circle_crop('./train_images/ffec9a18a3ce.png')
plt.imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))


# In[ ]:


spaceboy2 = circle_crop_v2('./train_images/ffec9a18a3ce.png')
plt.imshow(cv2.cvtColor(spaceboy2, cv2.COLOR_BGR2RGB))


# In[ ]:


from google.colab.patches import cv2_imshow

# Using cv2.imread() method 
img = cv2.imread('./train_images/ffec9a18a3ce.png') 
  
# Displaying the image 
plt.imshow(img)


# In[ ]:


xx= crop_image_from_gray(img,tol=7)
plt.imshow(xx)


# In[ ]:


def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    
    return img 


# In[ ]:


xxx = circle_crop('./train_images/ffec9a18a3ce.png')
cv2.imwrite('test_save.png', xxx)


# In[ ]:




