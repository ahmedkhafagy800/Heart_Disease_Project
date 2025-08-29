# refit_final_pipeline.py
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# نفس المحول المخصص المستخدم داخل البايبلاين
class ColumnSlicer(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = np.array(indices)
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices]

# تحميل مخرجات التدريب/الاختيار/الضبط
num = pd.read_csv('results/numeric_features.csv', header=None).iloc[:,0].tolist()
cat = pd.read_csv('results/categorical_features.csv', header=None).iloc[:,0].tolist()
sel = np.load('results/selected_feature_indices.npy')

# تحميل البيانات الخام (لازم لتدريب البايبلاين كاملاً)
raw = pd.read_csv('data/heart_disease.csv')
X_raw = raw[num + cat]
y_raw = raw['target']

# اختيار أفضل نموذج مُضبط وتحميله (يجب أن يكون مُدرّباً)
scores = pd.read_csv('results/tuned_auc_scores.csv', index_col=0, header=None).iloc[:,0]
best = scores.idxmax()
clf = joblib.load({
    'LogReg':'models/tuned_logreg.joblib',
    'RandomForest':'models/tuned_randomforest.joblib',
    'SVM_RBF':'models/tuned_svm.joblib'
}[best])

# خط أنابيب المعالجة نفسه المستخدم أثناء التدريب
num_tr = Pipeline([('impute', SimpleImputer(strategy='median')),
                   ('scale', StandardScaler())])
cat_tr = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                   ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num_tr, num), ('cat', cat_tr, cat)])

pipe = Pipeline([('preprocess', pre),
                 ('select', ColumnSlicer(sel)),
                 ('model', clf)])

# تدريب البايبلاين كاملاً لتصبح حالته fitted ثم الحفظ
pipe.fit(X_raw, y_raw)
Path('models').mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, 'models/final_model.pkl')
print('Refit pipeline saved to models/final_model.pkl using best =', best)
