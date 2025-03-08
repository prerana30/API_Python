from flask import Flask, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load dataset
file_path = "HisabKitab_data.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Feature Engineering
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag7'] = df['Sales'].shift(7)
df['Rolling_Avg_7'] = df['Sales'].rolling(window=7).mean()
df.fillna(method='bfill', inplace=True)

df['Festival'] = df['Festival'].astype('category').cat.codes
df['Season'] = df['Season'].astype('category').cat.codes

scaler = MinMaxScaler()
df['Average'] = scaler.fit_transform(df[['Average']])
df.dropna(subset=['Sales'], inplace=True)

# Define features and target
exog_features = ['Average', 'Festival', 'Season', 'Sales_Lag1', 'Sales_Lag7', 'Rolling_Avg_7']
target = 'Sales'

# Train-test split
train = df[df.index.year < 2023]
test = df[df.index.year == 2023]

train_exog = train[exog_features].fillna(method='ffill')
test_exog = test[exog_features].fillna(method='ffill')

# Auto-ARIMA to find best parameters
auto_model = auto_arima(train[target], exogenous=train_exog, seasonal=True, m=12, trace=True, suppress_warnings=True, stepwise=True, error_action='ignore', max_order=None, information_criterion='aic')

# Train SARIMAX model
sarimax_model = SARIMAX(train[target], exog=train_exog, order=auto_model.order, seasonal_order=auto_model.seasonal_order).fit()

# Predict Sales for 2023
test_predictions = sarimax_model.forecast(steps=len(test), exog=test_exog)
test['Predicted Sales'] = np.nan_to_num(test_predictions, nan=0.0)

# Forecast Sales for 2024
future_dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
future_data = pd.DataFrame(index=future_dates)
future_data['Average'] = df['Average'].mean()
future_data['Festival'] = df['Festival'].mode()[0]
future_data['Season'] = df['Season'].mode()[0]
future_data['Sales_Lag1'] = df['Sales'].iloc[-1]
future_data['Sales_Lag7'] = df['Sales'].iloc[-7]
future_data['Rolling_Avg_7'] = df['Sales'].rolling(window=7).mean().iloc[-1]

future_forecast = sarimax_model.forecast(steps=12, exog=future_data)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': future_forecast}).set_index('Date')

# Flask API Endpoints
@app.route('/actual_vs_predicted', methods=['GET'])
def actual_vs_predicted():
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train['Sales'], label="Training Data")
    plt.plot(test.index, test['Sales'], label="Actual Sales", color='red')
    plt.plot(test.index, test['Predicted Sales'], label="Predicted Sales", linestyle='dashed', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.title("Actual vs Predicted Sales (2023)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/future_sales_plot', methods=['GET'])
def future_sales_plot():
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Sales'], label="Historical Sales")
    plt.plot(forecast_df.index, forecast_df['Predicted Sales'], label="Forecasted Sales (2024)", linestyle='dashed', color='green')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.title("Future Sales Prediction (2024)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/forecasted_sales', methods=['GET'])
def forecasted_sales():
    # Convert Timestamp index to string
    forecast_json = forecast_df.reset_index()
    forecast_json['Date'] = forecast_json['Date'].astype(str)  # Convert Timestamp to string

    # Return JSON response
    return jsonify(forecast_json.to_dict(orient='records'))  # Convert to list of dictionaries


if __name__ == '__main__':
    app.run(debug=True)
