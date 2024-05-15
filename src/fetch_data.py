import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Download stock data
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")

# Moving Averages and other features
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['Momentum'] = data['Close'].diff(4)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
for i in range(1, 6):
    data[f'lag_{i}'] = data['Close'].shift(i)
data.dropna(inplace=True)

# Prepare data for model
feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'EMA_10', 'Momentum', 'RSI'] + [f'lag_{i}' for i in range(1, 6)]
X = data[feature_columns]
y = data['Close'].shift(-1)  # next day's close
X = X[:-1]
y = y.dropna()

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Square Error: {rmse}")
print(f"R^2 Score: {r2}")

# Saving results
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)
results.to_csv('AAPL_predictions.csv')

# Plotting
weekly_actual = results['Actual'].resample('W').mean()
weekly_predicted = results['Predicted'].resample('W').mean()
plt.figure(figsize=(14, 7))
plt.plot(weekly_actual.index, weekly_actual, label='Actual Price', marker='o', linestyle='-')
plt.plot(weekly_predicted.index, weekly_predicted, label='Predicted Price', color='red', linestyle='--', marker='x')
plt.title('AAPL Stock Price Prediction - Weekly Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
