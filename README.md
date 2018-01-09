# Google-Stock-Trend-Prediction
This is the code for a 4-layered recurrent neural network that predicts the trends in Google stock price based on previous 5 years of Stock data.
I built the model using the Keras library, which is built on top of Tensorflow and Theano. The inputs are numeric values of opening and closing stock price for 5 years from 2012 to 2016. The prediction is done for the Opening stock value for the month of January 2017. I used adam for stochastic optimization, and mean_squared_error as the loss function.
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
