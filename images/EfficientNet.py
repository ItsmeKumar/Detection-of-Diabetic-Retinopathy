#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive/') 


# In[5]:


get_ipython().system('pip install efficientnet')


# In[2]:


get_ipython().system('git clone https://github.com/Tony607/efficientnet_keras_transfer_learning')
# %cd efficientnet_keras_transfer_learning/


# In[6]:


from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input


# In[ ]:


import efficientnet.keras as efn 

base_model = efn.EfficientNetB0(weights='imagenet')


# In[13]:


from tensorflow.keras import models
from tensorflow.keras import layers

dropout_rate = 0.2
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation="softmax", name="fc_out"))


# In[14]:


base_model


# In[ ]:




