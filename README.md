Stock Price Prediction Using LSTM

Overview

This project implements a deep learning model to predict stock prices using Long Short-Term Memory (LSTM) networks. The model is trained using historical stock data and aims to forecast future stock prices based on past trends.

Features

Uses LSTM neural networks for time series forecasting.

Implements data preprocessing including feature scaling and time-step sequence creation.

Utilizes TensorFlow/Keras for model building and training.

Visualizes real vs. predicted stock prices.

Technologies Used

Python

NumPy

Pandas

Matplotlib

Scikit-Learn

TensorFlow/Keras

Dataset

The dataset used for training and testing is Google Stock Price data.

Training Data: Google_Stock_Price_Train.csv

Test Data: Google_Stock_Price_Test.csv

Installation

Prerequisites

Ensure you have Python installed along with the required dependencies. You can install them using:

Running the Project

Place the dataset files in the project directory.

Update the file paths in the script if necessary.

Run the script using:

Model Architecture

The LSTM model consists of:

4 LSTM layers with 50 units each.

Dropout layers to prevent overfitting.

A Dense output layer for predicting stock prices.

The Adam optimizer and mean squared error loss function.

Results

After training, the model predicts stock prices based on test data. The results are visualized using Matplotlib, comparing real vs. predicted stock prices.

Example Output



Future Improvements

Enhance accuracy using additional technical indicators.

Implement different architectures like GRUs or Transformers.

Deploy as a web application using Flask or FastAPI.

License

This project is open-source under the MIT License.

Author

[Suman Deep]

