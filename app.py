import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("AI Stock Market Analyzer & Predictor")

st.write("Analyze stock trends and predict future prices using LSTM Deep Learning.")


# -----------------------------
# USER INPUT
# -----------------------------

stock = st.text_input("Enter Stock Symbol", "AAPL")

start = st.date_input("Start Date", datetime(2015,1,1))
end = st.date_input("End Date", datetime.now())


# -----------------------------
# DOWNLOAD DATA
# -----------------------------

df = yf.download(stock, start=start, end=end)

if df.empty:
    st.error("No data found. Check the stock symbol.")
    st.stop()

st.subheader("Stock Data")
st.write(df.tail())


# -----------------------------
# STOCK STATISTICS
# -----------------------------

st.subheader("Stock Statistics")

st.write("Average Closing Price:", df['Close'].mean())
st.write("Highest Price:", df['High'].max())
st.write("Lowest Price:", df['Low'].min())


# -----------------------------
# PRICE CHART
# -----------------------------

st.subheader("Closing Price History")

fig = plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Closing Price")
st.pyplot(fig)


# -----------------------------
# MOVING AVERAGES
# -----------------------------

st.subheader("Moving Averages")

df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

fig2 = plt.figure(figsize=(12,5))
plt.plot(df['Close'], label='Close')
plt.plot(df['MA10'], label='10 Day MA')
plt.plot(df['MA20'], label='20 Day MA')
plt.plot(df['MA50'], label='50 Day MA')
plt.legend()
st.pyplot(fig2)


# -----------------------------
# LSTM PREDICTION
# -----------------------------

st.subheader("Stock Price Prediction using LSTM")

data = df[['Close']]
dataset = data.values

training_data_len = int(np.ceil(len(dataset) * 0.95))


# -----------------------------
# SCALE DATA
# -----------------------------

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# -----------------------------
# CREATE TRAINING DATA
# -----------------------------

train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# -----------------------------
# BUILD MODEL
# -----------------------------

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


# -----------------------------
# TRAIN MODEL
# -----------------------------

with st.spinner("Training AI model..."):
    model.fit(x_train, y_train, batch_size=1, epochs=1)


# -----------------------------
# TEST DATA
# -----------------------------

test_data = scaled_data[training_data_len - 60: , :]

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# -----------------------------
# PREDICTIONS
# -----------------------------

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

st.subheader("Model Performance")

st.write("Root Mean Squared Error (RMSE):", rmse)


# -----------------------------
# VISUALIZE PREDICTION
# -----------------------------

train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions

st.subheader("Prediction vs Actual")

fig3 = plt.figure(figsize=(12,5))

plt.plot(train['Close'], label="Train")
plt.plot(valid[['Close']], label="Actual")
plt.plot(valid[['Predictions']], label="Predicted")

plt.legend()

st.pyplot(fig3)


# -----------------------------
# AI SUGGESTION
# -----------------------------

st.subheader("AI Investment Suggestion")

last_actual = valid['Close'].tail(1).values[0]
last_predicted = valid['Predictions'].tail(1).values[0]

if last_predicted > last_actual:
    st.success("AI Suggestion: BUY 📈")
else:
    st.warning("AI Suggestion: SELL 📉")


st.success("Prediction Complete!")




