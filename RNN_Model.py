import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# ------------------ Part 1: Data Preprocessing ------------------

# Load Training Data
train_file = "Google_Stock_Price_Train.csv"
dataset_train = pd.read_csv(train_file)

# Extract 'Open' prices as training set
training_set = dataset_train.iloc[:, 1:2].values  # Assuming column 1 is 'Open'

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train, y_train = [], []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping for LSTM input (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ------------------ Part 2: Building the RNN ------------------

# Initializing the RNN
regressor = Sequential()

# Adding LSTM layers with Dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# ------------------ Part 3: Making Predictions ------------------

# Load Test Data
test_file = "Google_Stock_Price_Test.csv"
dataset_test = pd.read_csv(test_file)

# Extract real stock prices for comparison
real_stock_price = dataset_test.iloc[:, 1:2].values  # Assuming column 1 is 'Open'

# Prepare inputs for prediction
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)  # Normalize using the same scaler

# Create test dataset for prediction
X_test = []
for i in range(60, 80):  # Ensure index does not exceed dataset size
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict stock price
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # Convert back to original scale

# ------------------ Part 4: Visualization ------------------

plt.figure(figsize=(10, 5))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
