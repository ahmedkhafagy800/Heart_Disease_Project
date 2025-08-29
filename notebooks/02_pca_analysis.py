import numpy as np, matplotlib.pyplot as plt, joblib
from pathlib import Path
from sklearn.decomposition import PCA

Path('models').mkdir(parents=True, exist_ok=True)
X_train_p = np.load('results/X_train_preprocessed.npy')

pca = PCA(n_components=None, random_state=42).fit(X_train_p)
cum = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1,len(cum)+1), cum, marker='o'); plt.axhline(0.95, ls='--', c='r')
plt.xlabel('n_components'); plt.ylabel('cumulative explained variance')
plt.tight_layout(); plt.savefig('results/pca_cumulative_variance.png'); plt.close()

X2 = pca.transform(X_train_p)[:, :2]
plt.scatter(X2[:,0], X2[:,1], s=10, alpha=0.6)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
plt.savefig('results/pca_scatter_pc12.png'); plt.close()

joblib.dump(pca, 'models/pca.joblib')
