import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist

df = pd.read_csv("Final-data.csv")
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
k = 3 
kmeans = KMeans(n_clusters=k)
kmeans.fit(df_scaled)

df['cluster'] = kmeans.predict(df_scaled)

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

# pdist kullanılmadan yapılmış hali. // Not: pdist daha hızlı optimize ediyor.
"""
def calculate_dunn_index(data, kmeans):
    cluster_distances = pd.DataFrame(columns=range(kmeans.n_clusters), index=range(kmeans.n_clusters))
    for i in range(kmeans.n_clusters):
        for j in range(kmeans.n_clusters):
            if i != j:
                cluster_distances.iloc[i, j] = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j])

    min_cluster_distances = cluster_distances.min(axis=1)
    max_intra_cluster_distances = data.groupby('cluster').apply(lambda x: x.apply(np.linalg.norm, axis=1).max())

    dunn_index = min_cluster_distances.min() / max_intra_cluster_distances.max()

    return dunn_index
"""

def calculate_dunn_index(data, kmeans):
    cluster_distances = pd.DataFrame(columns=range(kmeans.n_clusters), index=range(kmeans.n_clusters))
    for i in range(kmeans.n_clusters):
        for j in range(kmeans.n_clusters):
            if i != j:
                cluster_distances.iloc[i, j] = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j])

    min_cluster_distances = cluster_distances.min(axis=1)
    
    max_intra_cluster_distances = []
    for i in range(kmeans.n_clusters):
        cluster_data = data[data['cluster'] == i]
        if not cluster_data.empty:
            distances = pdist(cluster_data.values, metric='euclidean')
            max_intra_cluster_distances.append(max(distances))

    dunn_index = min_cluster_distances.min() / max(max_intra_cluster_distances)

    return dunn_index

def calculate_bcss(kmeans, data):
    mean = data.mean().values

    bcss = 0
    for i in range(kmeans.n_clusters):
        cluster_mean = kmeans.cluster_centers_[i]
        bcss += len(data[data['cluster'] == i]) * euclidean_distance(mean, cluster_mean) ** 2
    return bcss

normalized_df = (df - df.min()) / (df.max() - df.min())

x_column = 'a3'
y_column = 'a4'

fig, ax = plt.subplots(figsize=(8, 6))
kmeans = KMeans(n_clusters=3).fit(normalized_df)
df['cluster'] = kmeans.labels_

colors = ['r', 'g', 'b']
for cluster in range(kmeans.n_clusters):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster}', marker='o', c=colors[cluster], s=50)

ax.legend(title='Cluster')
ax.grid(True)
ax.set_xlabel(x_column)
ax.set_ylabel(y_column)
ax.set_title('Clustering Results')

plt.savefig('plot.png')
plt.show()

wcss = kmeans.inertia_
bcss = calculate_bcss(kmeans, normalized_df)
dunn_index = calculate_dunn_index(df, kmeans)

for result_file in ['result_en.txt', 'result_tr.txt']:
    if os.path.exists(result_file):
        os.remove(result_file)

with open('result_en.txt', 'w') as f:
    for i, row in df.iterrows():
        f.write(f'Record {i+1}:\n')
        f.write(f'    Cluster {row["cluster"]}\n')

    for i in range(kmeans.n_clusters):
        cluster_size = len(df[df['cluster'] == i])
        f.write(f'Cluster {i}: {cluster_size} records\n')

    f.write(f'WCSS: {wcss:.2f}\n')
    f.write(f'BCSS: {bcss:.2f}\n')
    f.write(f'Dunn Index: {dunn_index:.2f}\n')

with open('result_tr.txt', 'w') as f:
    for i, row in df.iterrows():
        f.write(f'Kayit {i+1}:\n')
        f.write(f'    Kume {row["cluster"]}\n')

    for i in range(kmeans.n_clusters):
        cluster_size = len(df[df['cluster'] == i])
        f.write(f'Kume {i}: {cluster_size} kayit\n')

    f.write(f'WCSS: {wcss:.2f}\n')
    f.write(f'BCSS: {bcss:.2f}\n')
    f.write(f'Dunn Index: {dunn_index:.2f}\n')
