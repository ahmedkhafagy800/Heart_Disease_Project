from models.utils_transformers import ColumnSlicer
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

num = pd.read_csv('results/numeric_features.csv', header=None).iloc[:,0].tolist()
cat = pd.read_csv('results/categorical_features.csv', header=None).iloc[:,0].tolist()
sel = np.load('results/selected_feature_indices.npy')

num_tr = Pipeline([('imputer', SimpleImputer(strategy='median')),
                   ('scaler', StandardScaler())])
cat_tr = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                   ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num_tr, num), ('cat', cat_tr, cat)])

best = pd.read_csv('results/tuned_auc_scores.csv', index_col=0, header=None).iloc[:,0]
name = best.idxmax()
clf = joblib.load({'LogReg':'models/tuned_logreg.joblib',
                   'RandomForest':'models/tuned_randomforest.joblib',
                   'SVM_RBF':'models/tuned_svm.joblib'}[name])

class ColumnSlicer(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = np.array(indices)
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices]

pipe = Pipeline([('preprocess', pre), ('select', ColumnSlicer(sel)), ('model', clf)])
Path('models').mkdir(exist_ok=True, parents=True)
joblib.dump(pipe, 'models/final_model.pkl')
print(f'Saved models/final_model.pkl using best={name}')
