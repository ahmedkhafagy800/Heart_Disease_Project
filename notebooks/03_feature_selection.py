import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

X = np.load('results/X_train_preprocessed.npy')
y = pd.read_csv('results/y_train.csv').iloc[:,0]

rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)
rf_imp = rf.feature_importances_

lr = LogisticRegression(max_iter=2000, solver='liblinear')
rfe = RFE(lr, n_features_to_select=15).fit(X, y)
rfe_sup = rfe.support_

X_pos = X - X.min(axis=0) + 1e-6
chi = SelectKBest(chi2, k=15).fit(X_pos, y)
chi_sup = chi.get_support()

stab = rf_imp / rf_imp.max() + rfe_sup.astype(float) + chi_sup.astype(float)
idx = np.argsort(-stab)[:20]
np.save('results/selected_feature_indices.npy', idx)

plt.bar(range(len(idx)), stab[idx]); plt.xticks(range(len(idx)), [f'f{i}' for i in idx], rotation=90)
plt.tight_layout(); plt.savefig('results/feature_selection_stability.png'); plt.close()

joblib.dump({'rf':rf,'rfe':rfe,'chi':chi,'selected_idx':idx}, 'models/feature_selectors.joblib')
