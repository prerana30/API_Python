import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "HisabKitab_data.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)  # Ensure the index is sorted



# Feature Engineering
# Making a new column to predict as the data are dependent upon the past datas.
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag7'] = df['Sales'].shift(7)
df['Rolling_Avg_7'] = df['Sales'].rolling(window=7).mean()
df.fillna(method='bfill', inplace=True)  # Fill NaNs created by lagging and rolling

# Convert categorical variables to numerical
df['Festival'] = df['Festival'].astype('category').cat.codes
df['Season'] = df['Season'].astype('category').cat.codes

# Normalize 'Average' column using MinMaxScaler
# Without scaling, features with larger values might dominate the model's learning process, leading to biased results.
scaler = MinMaxScaler()
df['Average'] = scaler.fit_transform(df[['Average']])

# Ensure 'Sales' column exists and has no NaNs
df.dropna(subset=['Sales'], inplace=True)

# Define exogenous features and target
exog_features = ['Average', 'Festival', 'Season', 'Sales_Lag1', 'Sales_Lag7', 'Rolling_Avg_7']
target = 'Sales'

# Train-test split
train = df[df.index.year < 2023]  # Training: Before 2023
test = df[df.index.year == 2023]  # Testing: 2023

# Fill missing values in exogenous variables to prevent issues
train_exog = train[exog_features].fillna(method='ffill')
test_exog = test[exog_features].fillna(method='ffill')

# Auto-ARIMA to find best SARIMAX parameters
auto_model = auto_arima(
    train[target], 
    exogenous=train_exog, 
    seasonal=True, 
    m=12,  # Monthly seasonality
    trace=True, 
    suppress_warnings=True,
    stepwise=True,
    error_action='ignore',
    max_order=None,
    information_criterion='aic'
)

# Extract best parameters
best_order = auto_model.order
best_seasonal_order = auto_model.seasonal_order

# Train SARIMAX model
sarimax_model = SARIMAX(
    train[target], 
    exog=train_exog,
    order=best_order, 
    seasonal_order=best_seasonal_order
).fit()

# Predict Sales for 2023
test_predictions = sarimax_model.forecast(steps=len(test), exog=test_exog)

# Check if there are NaNs in predictions and fill them
test_predictions = np.nan_to_num(test_predictions, nan=0.0)

# Avoid SettingWithCopyWarning
test = test.copy()
test.loc[:, 'Predicted Sales'] = test_predictions

# Evaluate model performance
mae = np.mean(abs(test['Sales'] - test['Predicted Sales']))
rmse = np.sqrt(np.mean((test['Sales'] - test['Predicted Sales'])**2))
mape = np.mean(np.abs((test['Sales'] - test['Predicted Sales']) / test['Sales'])) * 100

# Print error metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot actual vs predicted sales
plt.figure(figsize=(12, 5))
plt.plot(train.index, train['Sales'], label="Training Data")
plt.plot(test.index, test['Sales'], label="Actual Sales", color='red')
plt.plot(test.index, test['Predicted Sales'], label="Predicted Sales", linestyle='dashed', color='blue')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.title("SARIMAX Model: Actual vs Predicted Sales (2023)")
plt.show()

# Forecast Sales for 2024
future_dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
future_data = pd.DataFrame(index=future_dates)

# Use historical average/mode values for exogenous features
future_data['Average'] = df['Average'].mean()
future_data['Festival'] = df['Festival'].mode()[0]
future_data['Season'] = df['Season'].mode()[0]
future_data['Sales_Lag1'] = df['Sales'].iloc[-1]  # Use the last observed sales value
future_data['Sales_Lag7'] = df['Sales'].iloc[-7]  # Use the sales value from 7 days ago
future_data['Rolling_Avg_7'] = df['Sales'].rolling(window=7).mean().iloc[-1]  # Use the last rolling average

# Predict sales for 2024
future_forecast = sarimax_model.forecast(steps=12, exog=future_data)

# Check if there are NaNs in future forecast and fill them
future_forecast = np.nan_to_num(future_forecast, nan=0.0)

# Create DataFrame for forecasted sales
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': future_forecast})
forecast_df.set_index('Date', inplace=True)

# Display forecast
print(forecast_df)

# Plot historical and future sales
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Sales'], label="Historical Sales")
plt.plot(forecast_df.index, forecast_df['Predicted Sales'], label="Forecasted Sales (2024)", linestyle='dashed', color='green')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.title("Future Sales Prediction using SARIMAX (2024)")
plt.show()

# Calculate residual errors
train['Residuals'] = train['Sales'] - sarimax_model.fittedvalues

# Compute MSE at each step
mse_loss = np.square(train['Residuals']).rolling(window=50).mean()

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(mse_loss, label="Training Loss (MSE)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("SARIMAX Training Loss Curve")
plt.legend()
plt.show()