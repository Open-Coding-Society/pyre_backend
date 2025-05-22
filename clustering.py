import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import hdbscan

# load data
df = pd.read_csv("fire_archive.csv", parse_dates=['acq_date'])

# feature renaming
df['month'] = df['acq_date'].dt.month
df['hour'] = df['acq_date'].dt.hour if 'acq_time' not in df.columns else df['acq_time'] // 100
df['is_day'] = df['daynight'].map({'D': 1, 'N': 0}) if 'daynight' in df.columns else 1

# features selection
features = ['latitude', 'longitude', 'brightness', 'frp', 'month', 'hour', 'is_day']
df_clean = df[features].dropna().sample(10000, random_state=42)

# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
clusters = clusterer.fit_predict(X_scaled)
df_clean['cluster'] = clusters

# save
df_clean.to_csv('fire_clusters_hdbscan.csv', index=False)

# visualize georgraphically
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='longitude', y='latitude', hue='cluster', palette='tab10', s=10)
plt.title("Fire Clusters (HDBSCAN)")
plt.tight_layout()
plt.show()

# t-sne plot
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
tsne_results = tsne.fit_transform(X_scaled)
df_clean['tsne-1'] = tsne_results[:, 0]
df_clean['tsne-2'] = tsne_results[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='tsne-1', y='tsne-2', hue='cluster', palette='tab10', s=10)
plt.title("t-SNE Projection of Fire Clusters")
plt.tight_layout()
plt.show()

# cluster analysis
print("\nCluster Counts:")
print(df_clean['cluster'].value_counts())

print("\nCluster Means:")
print(df_clean.groupby('cluster')[['brightness', 'frp', 'month', 'hour', 'is_day']].mean())

# cluster vs feature distributions/analysis
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean, x='cluster', y='month')
plt.title("Distribution of Fire Months by Cluster")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean, x='cluster', y='brightness')
plt.title("Brightness by Cluster")
plt.tight_layout()
plt.show()