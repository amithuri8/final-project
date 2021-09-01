#!/usr/bin/env python
# coding: utf-8

# #   Fashion MNIST

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA


# In[2]:


X_train = pd.read_csv(r'C:\Users\user\Desktop/fashion-mnist_test.csv')


# In[3]:


X_final_test = pd.read_csv(r'C:\Users\user\Desktop/fashion-mnist_train.csv')


# In[4]:


#lets see what the is..
X_train.head()


# # Labels
# 
# ## Each training and test example is assigned to one of the following labels:
# 
# ### 0 = T-shirt/top
# ### 1 = Trouser
# ### 2 = Pullover
# ### 3 = Dress
# ### 4 = Coat
# ### 5 = Sandal
# ### 6 = Shirt
# ### 7 = Sneaker
# ### 8 = Bag
# ### 9 = Ankle boot

# In[5]:


## lets check the size of our data.
print("Test size: " , X_final_test.shape)
print("Train size:" , X_train.shape)


# In[6]:


## delete duplicates.
X_final_test = X_final_test.drop_duplicates()
print(X_final_test.shape)

X_train = X_train.drop_duplicates()
print(X_train.shape)


# In[22]:


## check if there is null's data.

print(X_train.isnull().sum().sum())


# ### After i cleaned the data, ill start the visualization.

# # prepering our data to modeling...

# In[7]:


y_train = X_train["label"]
x_train = X_train.drop(["label"], axis=1)/255


# ### lets check if it really worked... :P

# In[8]:


x_train


# In[9]:


x_train , x_test , y_train , y_test = split(x_train,y_train ,train_size = 0.8 , shuffle = True , random_state = 1)


# ## using the models: 

# In[10]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[11]:


pca = PCA(n_components=16)
x_r = pca.fit_transform(x_train)

x_r.shape


# In[12]:


x_recovered = pca.inverse_transform(x_r)
x_recovered.shape


# In[13]:


#lets see the picturs before the PCA
label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
target = X_final_test[['label']].iloc[:, :]
plt.figure(figsize=(15,15))

for i in range(15):
    plt.subplot(3,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.array(X_final_test.drop(['label'],axis=1).iloc[i, :]).reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(label[target.label.iloc[i]])

# Show only the first 15 pictures:


# In[14]:


plt.figure(figsize=(15,15))

for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(x_r[i].reshape(4,4), cmap="gist_yarg")

plt.show

#after we doing it


# ## now i will check with 2 different calssification models:

# ### KNN

# In[15]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)
y_pred=knn.predict(x_test)
plt.figure(1, figsize= (10,10))
cm=confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
print("Confusion Matrix for KNN")
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
plt.close()


# In[16]:


knn.score(x_test, y_test)


# ### Gradient Boosting Modle

# In[17]:


gbc = GradientBoostingClassifier()


# In[18]:


gbc.fit(x_train, y_train)


# In[19]:


gbc.score(x_test, y_test)


# # To sum up...

# ### the Gradient Boosting have  has more successfuly.
# 
# ### with the biggest score: 86.05%
