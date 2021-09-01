#!/usr/bin/env python
# coding: utf-8

# In[142]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import os
import cv2
from skimage.io import imread
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA   
from sklearn.model_selection import train_test_split
import seaborn as sns
import time


# In[143]:


file_cat = r'C:\Users\user\Desktop/Cat/'


# In[144]:


## lets show our first 15 pictures of the cats.

train_images = [file_cat + i for i in os.listdir(file_cat)]

plt.figure(figsize=(10,10))

for i in range(15):
    plt.subplot(5,3,i+1)
    im = cv2.imread(train_images[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)


# In[145]:


## lets try to flip the pics of the cats:
plt.figure(figsize=(10,10))

for i in range(15):
    plt.subplot(5,3,i+1)
    im = cv2.imread(train_images[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.flip(im, 1)
    plt.imshow(im);


# In[146]:


file_dog = r'C:\Users\user\Desktop/Dog/'


# In[147]:


## lets show our first 15 pictures of the dogs.

t_images = [file_dog + i for i in os.listdir(file_dog)]

plt.figure(figsize=(10,10))

for i in range(15):
    plt.subplot(5,3,i+1)
    im2 = cv2.imread(t_images[i])
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    plt.imshow(im2)


# In[148]:


## now i'll flip the dogs pics:
plt.figure(figsize=(10,10))

for i in range(15):
    plt.subplot(5,3,i+1)
    im = cv2.imread(t_images[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.flip(im, 1)
    plt.imshow(im);


# In[149]:


filenames = os.listdir(file_dog)
classes = []
for filename in filenames:
    image_class = filename.split(".")[0]
    if image_class == "dog":
        classes.append(1)
    else:
        classes.append(0)


# In[150]:


df = pd.DataFrame({"filename": filenames, "category": classes})
df


# In[151]:


filenames = os.listdir(file_cat)
classes = []
for filename in filenames:
    image_class = filename.split(".")[0]
    if image_class == "cat":
        classes.append(1)
    else:
        classes.append(0)


# In[152]:


df2 = pd.DataFrame({"filename": filenames, "category": classes})
df2


# In[178]:


frames = [df, df2]
  
merge = pd.concat(frames)
merge


# In[ ]:




