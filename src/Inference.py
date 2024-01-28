import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px


np.random.seed(42)

# Load the pickled model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

ticker="msft"
start_date="2020-01-01"
end_date="2023-12-31"
interval="1d"

#download stock data 
stock=yf.download(ticker,start=start_date,end=end_date,interval=interval)
stock.reset_index(drop=False, inplace=True)

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

# RSI Function
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

# with predict RSI 
df_results['Predicted RSI'] = computeRSI(df_results['Predicted'], 14)

# Implement the trading strategy
initial_investment = 1000
num_shares_held = 0
cash_balance = initial_investment
portfolio_value = initial_investment
holding_shares = False

lower_rsi_threshold = 30
upper_rsi_threshold = 70

df_results['Signal'] = 0  # 0 for hold, 1 for buy, -1 for sell
df_results['Portfolio Value'] = 0  # Initialize Portfolio Value 

for i, row in df_results.iterrows():
    if row['Predicted RSI'] < lower_rsi_threshold and not holding_shares:
        # Buy signal and not already holding shares
        num_shares_to_buy = np.floor(cash_balance / row['Actual'])
        num_shares_held += num_shares_to_buy
        cash_balance -= num_shares_to_buy * row['Actual']
        df_results.at[i, 'Signal'] = 1
        holding_shares = True
    elif row['Predicted RSI'] > upper_rsi_threshold and holding_shares:
        # Sell signal and currently holding shares
        cash_balance += num_shares_held * row['Actual']
        num_shares_held = 0
        df_results.at[i, 'Signal'] = -1
        holding_shares = False

    portfolio_value = cash_balance + num_shares_held * row['Actual']
    df_results.at[i, 'Portfolio Value'] = portfolio_value

df_results_with_date = pd.merge(df_results, stock[['Date']], left_index=True, right_index=True, how='left')

# Mapping numeric values to their string representations 
signal_mapping= {0: 'Hold', 1: 'Buy', -1: 'Sell'}
df_results_final=df_results_with_date.copy()
df_results_final['Signal'] = df_results_with_date['Signal'].map(signal_mapping)

print(df_results_final.tail(15))

# Plotting Portfolio Value
fig_portfolio = px.line(df_results_with_date, x='Date', y='Portfolio Value',
                         labels={'value': 'Portfolio Value'}, title='Portfolio Value Over Time')

fig_portfolio.update_layout(
    xaxis_title='Time',
    yaxis_title='Portfolio Value',
    font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    template='plotly',
    height=400,  
    width=800
)

fig_portfolio.show()

#plot Signal

fig = px.line(df_results_final, x='Date', y=['Actual', 'Predicted'],
              labels={'value': 'Price'},
              title='Actual vs Predicted Prices with Buy/Sell Signals')

# Add arrows for buy and sell signals
for index, row in df_results_final.iterrows():
    if row['Signal'] == 'Buy':
        fig.add_annotation(x=row['Date'], y=row['Actual'],
                           text='Buy', arrowhead=2, showarrow=True, arrowcolor='green')
    elif row['Signal'] == 'Sell':
        fig.add_annotation(x=row['Date'], y=row['Actual'],
                           text='Sell', arrowhead=2, showarrow=True, arrowcolor='red')

fig.show()


