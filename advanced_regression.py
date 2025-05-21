import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Load and filter data
df = pd.read_csv("fire_archive.csv", parse_dates=['acq_date'])
df = df[(df['acq_date'].dt.year >= 2015) & (df['acq_date'].dt.year <= 2022)]

# Extract year and month
df['year_month'] = df['acq_date'].dt.to_period('M')

# Group by month
fires_per_month = df.groupby('year_month').size().reset_index(name='fire_count')
fires_per_month['year_month'] = fires_per_month['year_month'].astype(str)
fires_per_month['ds'] = pd.to_datetime(fires_per_month['year_month'])
fires_per_month['y'] = fires_per_month['fire_count']

# Drop unnecessary columns for Prophet
fires_prophet = fires_per_month[['ds', 'y']]

# Initialize and fit Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(fires_prophet)

# Forecast 12 months into the future
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot forecast with trend and uncertainty intervals
fig1 = model.plot(forecast)
plt.title("Forecasted Monthly Fire Count with Seasonality and Trend")
plt.xlabel("Date")
plt.ylabel("Fires per Month")
plt.tight_layout()
plt.show()

# Plot seasonality and trend components
fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()

# Optional: Plot actual vs predicted on historical data
merged = pd.merge(fires_prophet, forecast[['ds', 'yhat']], on='ds', how='left')
plt.figure(figsize=(14, 5))
sns.lineplot(data=merged, x='ds', y='y', label='Actual Fire Count')
sns.lineplot(data=merged, x='ds', y='yhat', color='red', label='Prophet Prediction')
plt.title("Actual vs Predicted Fire Counts (Prophet)")
plt.xlabel("Date")
plt.ylabel("Fire Count")
plt.tight_layout()
plt.show()
