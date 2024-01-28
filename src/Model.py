import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

ticker="msft"
start_date="2020-01-01"
end_date="2023-12-31"
interval="1d"

#download stock data 
stock=yf.download(ticker,start=start_date,end=end_date,interval=interval)
stock.reset_index(drop=False, inplace=True)


np.random.seed(42)

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append((seq, label))
    return np.array([s[0] for s in sequences]), np.array([s[1] for s in sequences])

seq_length = 14

data_df = stock.copy()

# Split the data based on dates
train_end_date = '2022-12-31'
train_data = data_df.loc[:train_end_date]
test_data = data_df.loc[train_end_date:]

# Extract data and create sequences for training and testing
scaler = MinMaxScaler(feature_range=(0, 1))

train_data_normalized = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
X_train, y_train = create_sequences(train_data_normalized, seq_length)

test_data_normalized = scaler.transform(test_data['Close'].values.reshape(-1, 1))
X_test, y_test = create_sequences(test_data_normalized, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=60, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)

# Reshape predictions to 2D array
predictions_reshaped = predictions.reshape(predictions.shape[0], predictions.shape[1])

# Inverse transform the predictions and actual values to the original scale
predictions_actual = scaler.inverse_transform(predictions_reshaped)
y_test_actual = scaler.inverse_transform(y_test.reshape(y_test.shape[0], y_test.shape[1]))

# Get actual dates from the preserved DataFrame, adjusting for sequence length
actual_dates = data_df.index[len(train_data):len(train_data) + len(X_test)]

# Create the DataFrame with actual dates
df_results = pd.DataFrame({'Actual': y_test_actual.flatten(), 'Predicted': predictions_actual.flatten()})
df_results = df_results.set_index(actual_dates)

print(df_results.tail(20))

pickle.dump(model, open('model.pkl', 'wb'))