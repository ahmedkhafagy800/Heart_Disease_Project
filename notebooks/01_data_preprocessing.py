import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt, seaborn as sns
from data.load_data import load_uci_heart

Path('results').mkdir(parents=True, exist_ok=True)
Path('data').mkdir(parents=True, exist_ok=True)

X, y = load_uci_heart()
df = X.copy(); df['target'] = y
df.to_csv('data/heart_disease.csv', index=False)

# EDA
df.hist(bins=20, figsize=(12,10)); plt.tight_layout()
plt.savefig('results/eda_histograms.png'); plt.close()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', square=True)
plt.tight_layout(); plt.savefig('results/eda_correlation_heatmap.png'); plt.close()

sns.boxplot(data=df[['age','trestbps','chol','thalach','oldpeak']])
plt.tight_layout(); plt.savefig('results/eda_boxplots.png'); plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

num = ['age','trestbps','chol','thalach','oldpeak']
cat = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

num_tr = Pipeline([('imputer', SimpleImputer(strategy='median')),
                   ('scaler', StandardScaler())])
cat_tr = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                   ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocess = ColumnTransformer([('num', num_tr, num), ('cat', cat_tr, cat)])

pd.Series(num).to_csv('results/numeric_features.csv', index=False, header=False)
pd.Series(cat).to_csv('results/categorical_features.csv', index=False, header=False)

X_train_p = preprocess.fit_transform(X_train, y_train)
X_test_p = preprocess.transform(X_test)
np.save('results/X_train_preprocessed.npy', X_train_p)
np.save('results/X_test_preprocessed.npy', X_test_p)
y_train.to_csv('results/y_train.csv', index=False)
y_test.to_csv('results/y_test.csv', index=False)
