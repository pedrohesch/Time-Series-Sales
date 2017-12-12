
# coding: utf-8

# # Predicting Time Series Forecast - Fast Food Sales
# # Using Idexes

# In[32]:

import pandas as pd
import matplotlib.pyplot as pl
get_ipython().magic('matplotlib inline')


# ### Importing data from Excel file. The data contains the monthly sale from 2014 to October 2017

# In[33]:

df = pd.read_excel('Faturamento+indices.xlsx')
df.head()


# ### Transforming the "Data" column to Index

# In[34]:

df.index = df['Data']
df.drop('Data', axis=1 , inplace=True)


# In[35]:

df.head()


# In[36]:

t = df.index.values
x = df.iloc[:,0]
pl.plot(x)
pl.show


# ### Creating the Output column (dependent column) for tranning data set, wich will be the "Vendas" shiftted by 1. It means the predicted sales to a month will be the sales of next month.
# 

# In[37]:

df['out'] = df['Vendas'].shift(1)
df.head()


# In[38]:

df1=df.copy()
df1.dropna(axis=0 , inplace=True)
df1.head()


# ### Normalizing the Data

# In[39]:

df2 = (df1-df1.mean())/(df1.max() - df1.min())
df2.head()


# ## Linear Regression

# ### Multi Variable (sales shiftted by 1 for output)
# ### Spliting the data into train and test
# ### Using seed to reproduce the variable selection
# ### MSE as a metric

# In[40]:

t = df2.index.values
x = df2.iloc[:,0:6]
y = df2.iloc[:,6]


# In[41]:

seed=7
from sklearn import model_selection
x_train , x_test , y_train , y_test = model_selection.train_test_split(x,y,test_size=0.2, random_state=seed)


# In[42]:

from sklearn.linear_model import LinearRegression
import numpy as np


# In[43]:

def rmse(y_real,y_pred):
    return np.sqrt(sum([ (m - n)**2 for m, n in zip(y_real,y_pred[:-1])]))/len(y_real)


# In[44]:

regr = LinearRegression()
regr.fit(x_train,y_train)


# In[45]:

mean_squared_error_train = rmse(y_train, regr.predict(x_train))
mean_squared_error_test = rmse(y_test, regr.predict(x_test))
print('Mean train squared error: %.5f' %  mean_squared_error_train)
print('Mean test squared error: %.5f' %  mean_squared_error_test)


# In[46]:

y_pred = regr.predict(x)
pl.plot(t,y,'r-')
pl.plot(t,y_pred)
pl.show()


# ### Creating the new input data to predict

# In[49]:

p= df.iloc[0,0:6]
p


# In[51]:

p= df.iloc[0,0:6]
p= (p-df1.iloc[:,0:6].mean())/(df1.iloc[:,0:6].max() - df1.iloc[:,0:6].min())
p= p.reshape(1,-1)
p


# In[52]:

prev = regr.predict(p)


# ### Result (prediction for next month)

# In[53]:

prev = (prev*(df1["out"].max() - df1["out"].min()))+df1["out"].mean()
prev


# In[ ]:



