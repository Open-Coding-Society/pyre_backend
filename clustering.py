from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("fire_archive.csv", parse_dates=['acq_date'])

X = df[['latitude', 'longitude']].dropna().sample(10000)

# Normalize for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
X['cluster'] = db.fit_predict(X_scaled)

# Save
X[['latitude', 'longitude', 'cluster']].to_csv('fire_dbscan_clusters.csv', index=False)

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X, x='longitude', y='latitude', hue='cluster', palette='tab10', s=10)
plt.title("Fire Clusters (DBSCAN)")
plt.tight_layout()
plt.show()
