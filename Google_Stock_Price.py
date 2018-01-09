# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:50:52 2018

@author: sumit
"""

# =============================================================================
# RECURRENT NEURAL NETWORK
# =============================================================================

## Part 1: DATA PREPROCESSING

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the train set data
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# Creating a np array for the training data
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a datastructure with 60 time steps and 1 output:

# x_train contains 60 previous stock prices before a particular financial day.
# y_train contains stock price for the next financial day. 
# This is done for every time = t

x_train = []
y_train = []

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping to make the input compatible for RNN
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))


## Part 2: BUILDING THE RNN
# Importing keras library and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN:
regressor = Sequential()

# Adding the first LSTM layer with some dropout regularization
# (This will be a stacked LSTM with several LSTM layers)
regressor.add(LSTM(units=50, return_sequences = True, 
                   input_shape= (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM layer with some dropout regularization
# No input shape required from second layer
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer with some dropout regularization
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer with some dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
# Since the output layer is a Fully connected layer, 
# we will use the Dense function.
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

## Part 3: Making the predictions and visualizing the results
# Getting real stock price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

# Predicting stock price of 2017
# (Since the RNN model was trained on the scaled inputs, we will scale the input)
dataset_total = pd.concat((dataset_train['Open'], 
                            dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)- len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # (Since we have already fitted earlier 
 # we use only transform)
 
x_test = [] 
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
  
# Visualizing the results
plt.plot(real_stock_price, color ="red", label = "Real Google stock price")
plt.plot(predicted_stock_price, color="green", 
         label="Predicted Google stock price")
plt.title("Google stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
 
# Here we are more interested in checking the trend of the Google stock price
# and not much on the accuracy of prediction.
# But let us try to compute the accuracy of the model anyway:
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse
 
# Computing the absolute error for this case
abs_error = rmse/800 # we divide by the range of google stock price values
abs_error

# =============================================================================
# Steps that can be taken to further improve the model:

# 1. Getting more training data: We trained our model on the past 5 years 
#    of the Google Stock Price but it would be even better to train it on 
#    the past 10 years.
# 2. Increasing the number of timesteps: the model remembered the stock 
#    prices from the 60 previous financial days to predict the stock price 
#    of the next day. That’s because we chose a number of 
#    60 timesteps (3 months). You could try to increase the number of 
#    timesteps, by choosing for example 120 timesteps (6 months).
# 3. Adding some other indicators: if you have the financial instinct that 
#    the stock price of some other companies might be correlated to the one 
#    of Google, you could add this other stock price as a new indicator in 
#    the training data.
# 4. Adding more LSTM layers: we built a RNN with four LSTM layers 
#    but more layers can be added depending on buisness intuition.
# 5. Adding more neurons in the LSTM layers: we highlighted the fact that 
#    we needed a high number of neurones in the LSTM layers to respond better 
#    to the complexity of the problem and we chose to include 50 neurones in  
#    each of our 4 LSTM layers. You could try an architecture with even more 
#    neurones in each of the 4 (or more) LSTM layers.
# =============================================================================


