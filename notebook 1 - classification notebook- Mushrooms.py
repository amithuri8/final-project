#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# # Data set of mushrooms

# In[22]:


df = pd.read_csv(r'C:\Users\user\Desktop/mushrooms.csv')
df

## uploading the csv file...


# ## Overview
# classes: edible = e, poisonous=p
# 
# cap-shape: bell = b, conical = c,convex = x, flat = f, knobbed = k, sunken = s
# 
# cap-surface: fibrous = f, grooves = g, scaly = y, smooth = s
# 
# cap-color: brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
# 
# bruises: yes = t, no = f
# 
# odor: almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
# 
# gill-attachment: attached = a, descending = d, free = f, notched = n
# 
# gill-spacing: close = c, crowded = w, distant = d
# 
# gill-size: broad = b, narrow = n
# 
# gill-color: black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w ,yellow = y
# 
# stalk-shape: enlarging = e, tapering = t
# 
# stalk-rootbulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
# 
# stalk-surface-above-ring: fibrous = f, scaly = y, silky = k, smooth = s
# 
# stalk-surface-below-ring: fibrous = f, scaly = y, silky = k, smooth = s
# 
# stalk-color-above-ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
# 
# stalk-color-below-ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
# 
# veil-type: partial = p, universal = u
# 
# veil-color: brown = n, orange = o, white = w, yellow = y
# 
# ring-number: none = n, one = o, two = t
# 
# ring-type: cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
# 
# spore-print-color: black = k, brown = n, buff = b, chocolate = h, green = r, orange = o,purple = u, white = w, yellow = y
# 
# population: abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
# 
# habitat: grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d

# In[23]:


df.groupby(['cap-color', 'class'])


# In[ ]:





# In[24]:


df = df.drop_duplicates()
df

#Dropping the duplicated rows if there is...


# #### checking the data...

# In[25]:


print(df.isnull().sum())

## now we checking if there is an unnecessary data.


# In[26]:


df.shape

##Checking how many rows and columns we have.


# In[27]:


df['class'].value_counts()

#checking how many Edible, Poisonous we have in the data set.


# In[28]:


sns.catplot(x="cap-color",hue="class", data=df, kind="count",height=8, aspect=.8, palette =['#af1222','#6A9320']);
plt.title("poision relative to edible")

#The correlation between cap color and Edible or Poisonous


# In[29]:


plt.figure(figsize = (6,6))
plt.pie(df['class'].value_counts(), startangle = 90, autopct = '%.1f', labels = ['Edible', 'Poisonous'])
plt.show()

## a pie diag that shows the precents of each category in 'class'.


# In[30]:


r,c = 7,3
fig,axes = plt.subplots(r,c,figsize=(15,30))

X = df.columns[1:]
ctr = 0

for i in range(r):
    for j in range(c):
        
        sns.countplot(ax=axes[i][j] ,x = df[X[ctr]],hue=df['class'])
        ctr += 1

plt.subplots_adjust(wspace=0.4, hspace=0.4) 

## now we show each category relative to the predict parameter. 
## i chose to show it in a column diagram.


# ## Encoding categorical variables numerically for classification

# In[31]:


categorical_df = df.select_dtypes(include=['object'])
categorical_df.columns


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

categorical_df = categorical_df.apply(enc.fit_transform)
categorical_df.head()


df = df.drop(categorical_df.columns, axis=1)
df = pd.concat([df, categorical_df], axis=1)
df


# ### starting the classification models...

#  Define x and y

# In[32]:


x = df.drop('class', axis=1)
y = df['class']


# In[33]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# ### Dummy model

# In[34]:


from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train,y_train)
pred_dummy = dummy.predict(x_test)
Dummy_res = accuracy_score(y_test, pred_dummy)
Dummy_res


# In[35]:


print(classification_report(y_test, pred_dummy))


# In[36]:


print(confusion_matrix(y_test,pred_dummy))
#tn  fp
#fn  tp


# ### SVC model

# In[40]:


from sklearn.svm import SVC

svc =  SVC(kernel="linear", C=0.025)
svc.fit(x, y)
pred_svc = svc.predict(x_test)

SVC_res = accuracy_score(y_test, pred_svc)
SVC_res


# In[ ]:


print(classification_report(y_test, pred_svc ))


# In[41]:


print(confusion_matrix(y_test,pred_svc))
#tn  fp
#fn  tp


# ### Logistic Regression model

# In[42]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train, y_train)
pred_logistic = logreg.predict(x_test)

LogReg_res = accuracy_score(y_test, pred_logistic)
LogReg_res


# In[43]:


print(classification_report(y_test, pred_logistic))


# In[44]:


print(confusion_matrix(y_test,pred_logistic))
#tn  fp
#fn  tp


# #### K - Nearest Neighbors model

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for i in range(2, 51):
    classifier = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)
    
    # Make the prediction
    y_pred = classifier.predict(x_test)

    if i==7:
        KNN_res = accuracy_score(y_test, y_pred)
        
    print('Accuracy in k=' +str(i)+ ' is: ' + str(accuracy_score(y_test, y_pred)))


# ### Question for roe...

# In[46]:


## The mean of the results in KNN model (from 0 to 50) is 0.979815422477441, 
## is it correct to say that the accuracy of the 'KNN' model is that meanning of the results?

score_list = []
for each in range(1,51):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))
plt.plot(range(1,51), score_list)
plt.xlabel("k's value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# ## knn- conclusion...
# I would choose k=7 for the best K that could have been chosen. 
# 
# The reason is when k=7, it gives me the best result, maybe k=2 gives us more accuracy but its a little to much relative to our data size.
# 
# (when k equals 2, 3 or 4 the result got the same value. but, the k is to little relatively to the data.) 

# # For conclusion 

# In[47]:


# x-coordinates of left sides of bars 
count = [1, 2, 3, 4] 

# heights of bars 
results = [Dummy_res, SVC_res, LogReg_res, KNN_res]

# labels for bars 
names = ['Dummy_res', 'SVC_res', 'LogReg_res', 'KNN_res'] 

# plotting a bar chart 
plt.bar(count, results, tick_label = names, color = ['yellow', 'green', 'red', 'brown']) 

# naming the x axis 
plt.xlabel('result') 

# naming the y axis 
plt.ylabel('which model') 
# plot title 

plt.title('The results of the models:') 

# function to show the plot 
plt.show() 


#  i would choose the 'KNN' model. that because it gives me the best result to my predict.

# # Part 2: Semester B:

# In[48]:


from sklearn.decomposition import PCA


# In[49]:


pca = PCA(n_components= 0.9)


# In[51]:


pca.fit(x_train)


# In[52]:


pca.n_components_


# In[53]:


train_pca = pca.transform(x_train)
testpca = pca.transform(x_test)


# In[56]:


from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score


# In[63]:


import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
adac = AdaBoostClassifier(
algorithm="SAMME.R", learning_rate=0.01)
adac.fit(train_pca , y_train)
scores = cross_val_score(adac, testpca, y_test, cv=5)
print("AdaBoost Classifier score: ", scores.mean())


# In[ ]:





# In[ ]:




