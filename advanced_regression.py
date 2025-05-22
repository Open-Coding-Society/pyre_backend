import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from matplotlib.animation import FuncAnimation

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

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, len(fires_per_month))
ax.set_ylim(0, fires_per_month['fire_count'].max() * 1.1)
ax.set_title("Monthly Fire Counts Over Time (Animated)")
ax.set_xlabel("Month")
ax.set_ylabel("Fire Count")

line, = ax.plot([], [], lw=2, color='red')
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

x_data = []
y_data = []

def update(frame):
    x_data.append(fires_per_month['year_month'][frame])
    y_data.append(fires_per_month['fire_count'][frame])
    line.set_data(range(len(x_data)), y_data)
    text.set_text(f"Month: {fires_per_month['year_month'][frame]}\nFires: {y_data[-1]}")
    ax.set_xticks(range(len(x_data)))
    ax.set_xticklabels(x_data, rotation=45, ha='right', fontsize=8)
    return line, text

ani = FuncAnimation(fig, update, frames=len(fires_per_month), interval=200, repeat=False)
plt.tight_layout()
plt.show()