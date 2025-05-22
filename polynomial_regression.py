import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


df = pd.read_csv("fire_archive.csv", parse_dates=['acq_date'])

# Optional: Filter years
df = df[(df['acq_date'].dt.year >= 2015) & (df['acq_date'].dt.year <= 2022)]

# Extract year/month
df['year'] = df['acq_date'].dt.year
df['month'] = df['acq_date'].dt.month
df['year_month'] = df['acq_date'].dt.to_period('M')

fires_per_month = df.groupby('year_month').size().reset_index(name='fire_count')
fires_per_month['year_month'] = fires_per_month['year_month'].astype(str)

fires_per_month['timestamp'] = pd.to_datetime(fires_per_month['year_month']).astype(int) / 10**9  # Unix time

X = fires_per_month['timestamp'].values.reshape(-1, 1)
y = fires_per_month['fire_count'].values

poly = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly.fit(X, y)

fires_per_month['poly_pred'] = poly.predict(X)

sns.scatterplot(x='year_month', y='fire_count', data=fires_per_month, label='Actual')
sns.lineplot(x='year_month', y='poly_pred', data=fires_per_month, color='green', label='Polynomial Trend')
plt.xticks(rotation=45)
plt.title("Polynomial Regression (Degree 4) Trend")
plt.legend()
plt.tight_layout()
plt.show()
