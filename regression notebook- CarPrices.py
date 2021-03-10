#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[3]:


df = pd.read_csv(r'C:\Users\user\Desktop/CarPrice_Assignment.csv')
df

## uploading the csv file...


# In[4]:


df.shape

##checking the size of the original dataset


# #### checking the data...

# In[5]:


df.duplicated().any()

#checking if there is some doplicates by getting a boolean answer.


# In[6]:


df = df.drop_duplicates()
df.duplicated().any()

#Dropping the duplicated rows if there is...


# In[7]:


df.drop(['torque'],axis=1, inplace = True)
df.head()

# i decided to delete this column because i didnt rlly know how to convert this data to a numerical data, and i dont think that
# it rlly a parameter that we can use.


# In[8]:


df.isnull().sum()

## now we checking if there is an unnecessary data.


# In[9]:


df.isnull().sum() / df.shape[0] * 100

# checks what percentage they constitute.


# #### Because i have got a lot of data and this 'null' data is only on 3% i decide to delete the data.

# In[10]:


df.dropna(axis=0, inplace=True)
df.isnull().any()

# delete the null data and checks for sure :)


# In[11]:


df.shape

#checking agian the size of the data after the deleting...


# ### Prepares the data for the prediction process :

# In[12]:


df


# for doing our prediction we need to convert all our data to float data (numerical data). .\
# that becaue the commends we using cannot read a strings data.

# #### Converting the data into float format since they are numerical data

# In[40]:


#lets check wich data is 'int' or 'float' and which is 'object'
df.info()


# In[13]:


#converting our string to int by deleting the strings and convert the numbers to a int Dtype.

df['mileage'] = df['mileage'].str.strip('kmpl').str.strip('km/kg')
df['engine'] = df['engine'].str.strip('CC')
df['max_power'] = df['max_power'].str.strip('bhp').str.strip()

df.head()

#deleting the strings from those collumns.


# In[14]:


df.info()

#check which data is an object and which is a num.


# In[15]:


df['mileage'] = pd.to_numeric(df['mileage'])
df['engine'] = pd.to_numeric(df['engine'])
df['max_power'] = pd.to_numeric(df['max_power'])
df.head()


# In[16]:


df.info()

#check if the data rlly changed.


# In[17]:


get_num = ['fuel', 'seller_type', 'transmission', 'owner']
dummies = pd.get_dummies(df[get_num], drop_first= True)


# In[18]:


df_final = pd.concat([df, dummies],axis = 1)
df_final.drop(['fuel', 'seller_type', 'transmission','name', 'owner'], axis = 1, inplace = True)

df_final.head()

#every data that has been as string, it convert it to a '0' or '1' by adding all of the options that every collmun get.


# ### start the visualization

# In[19]:


df.groupby('year').mean()['selling_price'].plot(kind = 'line', figsize=(10,6))
plt.ylabel('selling_price');


# we can see that if the year increases the selling price is getting increase too.

# In[20]:


# plotting categorical variables vs target variable selling price
plt.figure(figsize=[20,10])
plt.subplot(2,2,1)

sns.barplot(df.owner, df.selling_price)
plt.subplot(2,2,2)

sns.barplot(df.fuel, df.selling_price)
plt.subplot(2,2,3)

sns.barplot(df.seller_type, df.selling_price)
plt.subplot(2,2,4)

sns.barplot(df.transmission, df.selling_price)

plt.show()


# In[21]:


# km_driven vs selling_price
plt.figure(figsize=[14,8])
sns.scatterplot(df.km_driven, df.selling_price)
plt.show()


# As we can see km_driven has 2 outliers, lets delete them from the dataset. .\
# As we can see that if the km_driven increases the selling price is getting decreased.

# In[22]:


df = df[df.km_driven < 1000000]

#deleting...


# In[23]:


#checks if it really deleted

plt.figure(figsize=[14,8])
sns.scatterplot(df.km_driven, df.selling_price)
plt.show()


# In[24]:


# engine vs selling_price

plt.figure(figsize=[12,8])
sns.scatterplot(df.engine, df.selling_price)
plt.show()


# agian, we can see that we got 2 outliers, lets delete them from the dataset.

# In[25]:


df = df[df.engine < 3400]

#deleting...


# In[26]:


plt.figure(figsize=[12,8])
sns.scatterplot(df.engine, df.selling_price)
plt.show()

#checks if it really deleted


# In[27]:


# mileage vs selling_price

plt.figure(figsize=[12,8])
sns.scatterplot(df.mileage, df.selling_price)
plt.show()


# In[28]:


# seats vs selling_price
plt.figure(figsize=[12,8])
sns.scatterplot(df.seats, df.selling_price)
plt.show()


# In[29]:


# looking at pair plot for numerical data
df_final.corr()


# In[30]:


df_final.dropna(axis=0, inplace=True)


# ### Correlation table:

# In[31]:


plt.figure(figsize = [15,8])
sns.heatmap(df_final.corr(), annot = True, cmap = 'RdYlGn')
plt.show()


# ## Let's start the process of predicting

# In[32]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[33]:


x = df_final.drop('selling_price', axis=1)
y = df_final['selling_price']


# In[34]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)


# ### Linear Regression model

# In[35]:


from sklearn.metrics import r2_score, mean_squared_error

#r2 matric- This metric is not well-defined for single samples and will return a NaN value if n_samples is less than two.

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(x)
r2score = r2_score(y, y_pred)
r2score


# ### Random Forest Regressor

# In[36]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred1 = rf.predict(x)
r2score1 = r2_score(y, y_pred1)
r2score1

# taking the sum of the predicted 


# ### Gradient Boosting Regressor

# In[37]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
y_pred2 = gbr.predict(x)
r2score2 = r2_score(y, y_pred2)
r2score2


# ### for conclusion

# In[38]:


# x-coordinates of left sides of bars 
count = [1, 2, 3] 

# heights of bars 
results = [r2score2, r2score1, r2score]

# labels for bars 
names = ['gbr', 'rf', 'lr'] 

# plotting a bar chart 
plt.bar(count, results, tick_label = names, color = ['yellow', 'green', 'red']) 

# naming the x axis 
plt.xlabel('result') 

# naming the y axis 
plt.ylabel('which model') 
# plot title 

plt.title('The results of the models:') 

# function to show the plot 
plt.show() 


# i would choose the 'Random Forest Regressor' model. that because it gives me the best result to my predict.
