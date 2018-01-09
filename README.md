# Google-Stock-Trend-Prediction
This is the code for a 4-layered recurrent neural network that predicts the trends in Google stock price based on previous 5 years of Stock data.
I built the model using the Keras library, which is built on top of Tensorflow and Theano. The inputs are numeric values of opening and closing stock price for 5 years from 2012 to 2016. 
The prediction is done for the Opening stock value for the month of January 2017. I used adam for stochastic optimization, and mean_squared_error as the loss function.
# Dependencies
● tensorflow ● keras ● numpy ● pandas ● scikit-learn
# Dataset
| Variable       | Definition   | 
| ------------- |:-------------:|
| Date      | Date of stock price record |
| Open      | Opening value of stock price     |
| High | Highest value of stock price on that day   |
| Low | Lowest value of stock price on that day |
| Close | Closing value of stock price |
| Volume | Total Volume of trade |

# Usage
Run Google_Stock_Price.py in terminal to see the network in training. I have used Spyder from Anaconda to script and visualize the code.

```
Epoch 95/100
1198/1198 [==============================] - 7s - loss: 0.0015     
Epoch 96/100
1198/1198 [==============================] - 7s - loss: 0.0014     
Epoch 97/100
1198/1198 [==============================] - 7s - loss: 0.0015         
Epoch 98/100
1198/1198 [==============================] - 7s - loss: 0.0016     
Epoch 99/100
1198/1198 [==============================] - 7s - loss: 0.0013         
Epoch 100/100
1198/1198 [==============================] - 7s - loss: 0.0014     

```
![alt text](https://github.com/sumitpgupta/Google-Stock-Trend-Prediction/blob/master/Real%20Vs%20Predicted%20Stock%20price.png)

# Steps to further improve the model:
Steps that can be taken to further improve the model:

1. Getting more training data: We trained our model on the past 5 years 
    of the Google Stock Price but it would be even better to train it on 
    the past 10 years.
 2. Increasing the number of timesteps: the model remembered the stock 
    prices from the 60 previous financial days to predict the stock price 
    of the next day. That’s because we chose a number of 
    60 timesteps (3 months). You could try to increase the number of 
    timesteps, by choosing for example 120 timesteps (6 months).
 3. Adding some other indicators: if you have the financial instinct that 
    the stock price of some other companies might be correlated to the one 
    of Google, you could add this other stock price as a new indicator in 
    the training data.
 4. Adding more LSTM layers: we built a RNN with four LSTM layers 
    but more layers can be added depending on buisness intuition.
 5. Adding more neurons in the LSTM layers: we highlighted the fact that 
    we needed a high number of neurones in the LSTM layers to respond better 
    to the complexity of the problem and we chose to include 50 neurones in  
    each of our 4 LSTM layers. You could try an architecture with even more 
    neurones in each of the 4 (or more) LSTM layers.
