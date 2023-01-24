#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LSTM model for stock price analysis


# In[2]:


#importing the necessary packages
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn. preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[5]:


#read the file
df = pd.read_csv('Price.csv')
df


# In[6]:


#Getting the number of rows and columns
df.shape


# In[7]:


df.head()


# In[8]:


#Changing the index to Date column
df = df.set_index('Date')
df


# In[9]:


#Visualizing the closing price
plt.figure(figsize=(16,8))
plt.title('Close price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# In[10]:


#Create a new dataframe with only the "Close column"
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len=math.ceil( len(dataset)*.8)

training_data_len


# In[11]:


#Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[12]:


#Create the training dataset
#Create the scaled training dataset
train_data = scaled_data[0:training_data_len,:]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[13]:


#Convert the x_train and y_train to numpy arrays
x_train,y_train = np.array(x_train),np.array(y_train)


# In[14]:


#Reshape the data(The LSTM model expects 3 dimensions)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[15]:


#Building the model
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[16]:


#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[17]:


#Train the model
model.fit(x_train,y_train,batch_size=1, epochs=1)


# In[18]:


#Create the testing dataset
#Create a new array containing scaled values
test_data = scaled_data[training_data_len-60:,:]
#Create the datases x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[19]:


#Convert the data to a numpy array
x_test = np.array(x_test)


# In[20]:


#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1 ))


# In[21]:


#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[22]:


#Get the root mean squared error(RMSE)
rmse = np.sqrt( np.mean(predictions-y_test)**2 )
rmse


# In[26]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()


# In[27]:


#show the valid and predicted prices
valid

