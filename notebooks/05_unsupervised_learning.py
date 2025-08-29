import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, dendrogram

out = Path('results')
Xtr = np.load(out/'X_train_preprocessed.npy'); Xte = np.load(out/'X_test_preprocessed.npy')
ytr = pd.read_csv(out/'y_train.csv').iloc[:,0]; yte = pd.read_csv(out/'y_test.csv').iloc[:,0]
X = np.vstack([Xtr, Xte]); y = np.concatenate([ytr.values, yte.values])

Ks = range(2,9); inertias = []; sils = []
for k in Ks:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    lab = km.fit_predict(X)
    inertias.append(km.inertia_); sils.append(silhouette_score(X, lab))
plt.plot(list(Ks), inertias, marker='o'); plt.xlabel('k'); plt.ylabel('Inertia')
plt.tight_layout(); plt.savefig(out/'kmeans_elbow.png'); plt.close()
plt.plot(list(Ks), sils, marker='o'); plt.xlabel('k'); plt.ylabel('Silhouette')
plt.tight_layout(); plt.savefig(out/'kmeans_silhouette.png'); plt.close()

km2 = KMeans(n_clusters=2, n_init='auto', random_state=42)
lab2 = km2.fit_predict(X)
ari = adjusted_rand_score(y, lab2); nmi = normalized_mutual_info_score(y, lab2)
with open(out/'clustering_metrics.txt','w') as f:
    f.write(f'KMeans k=2 ARI={ari:.3f}, NMI={nmi:.3f}\n')

Z = linkage(X[:200], method='ward')
plt.figure(figsize=(10,4)); dendrogram(Z, truncate_mode='lastp', p=30)
plt.tight_layout(); plt.savefig(out/'hierarchical_dendrogram.png'); plt.close()
