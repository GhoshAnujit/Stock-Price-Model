# Stock Price Prediction with LSTM

## Overview

This project was conducted during my 2-month Data Science internship at Corizo Ed.Tech. The primary objective was to predict stock prices using historical stock price data provided by Corizo. To achieve this, we utilized a Long Short-Term Memory (LSTM) model from the Keras library. The model was trained on a labeled dataset and evaluated by comparing its predictions with the actual closing prices.

## Dataset

The dataset used for this project consists of historical stock price data. It includes features such as Open, High, Low, and Close prices. The data was split into training and testing sets to facilitate model training and evaluation.

## LSTM Model

For this task, we employed a Long Short-Term Memory (LSTM) neural network model. LSTM networks are well-suited for time series data like stock prices because they can capture sequential dependencies in the data. Below is an overview of the LSTM model architecture:

- LSTM Layer 1: 50 units, return sequences=True, input shape=(X_train.shape[1], 1)
- LSTM Layer 2: 50 units, return sequences=False
- Dense Layer 1: 25 units
- Dense Layer 2: 1 unit (output)

## Training and Testing

The dataset was divided into training and testing sets. The model was trained on the training data, and predictions were generated using the testing data. To evaluate the model's performance, we compared the predicted stock prices with the actual closing prices.

## Results

**Closing Prices Graph**

![Closing_Prices_graph}(Close_prices.png)

**Predicted Prices vs. Testing Values**

![Predicted_Prices_Graph}(Predictions.png)

## Conclusion

In this project, we successfully applied an LSTM model to predict stock prices based on historical data. The provided graphs of closing prices and predicted vs. actual prices offer insights into the model's performance.

## Dependencies

Python
Pandas
Numpy
Scikit-Learn
Keras
Matplotlib


## Acknowledgments

I thank CORIZO Ed.Tech. for providing me an opportunity to work on this project and learn how to deal with time series data.
