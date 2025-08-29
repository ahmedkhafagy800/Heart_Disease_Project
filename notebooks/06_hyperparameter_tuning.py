import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform, randint

out = Path('results')
Xtr = np.load(out/'X_train_preprocessed.npy'); Xte = np.load(out/'X_test_preprocessed.npy')
ytr = pd.read_csv(out/'y_train.csv').iloc[:,0]; yte = pd.read_csv(out/'y_test.csv').iloc[:,0]
sel = np.load(out/'selected_feature_indices.npy')
Xtr = Xtr[:, sel]; Xte = Xte[:, sel]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_lr = GridSearchCV(LogisticRegression(max_iter=5000, solver='liblinear'),
                       {'C':[0.01,0.1,1,10,100],'penalty':['l1','l2']},
                       scoring='roc_auc', cv=cv, n_jobs=-1).fit(Xtr, ytr)
best_lr = grid_lr.best_estimator_

rand_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
    {'n_estimators': randint(200,700),'max_depth': randint(3,12),
     'min_samples_split': randint(2,10),'min_samples_leaf': randint(1,5),
     'max_features':['sqrt','log2', None]},
    n_iter=40, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1).fit(Xtr, ytr)
best_rf = rand_rf.best_estimator_

rand_svm = RandomizedSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
    {'C': loguniform(1e-2, 1e3), 'gamma': loguniform(1e-4, 1e0)},
    n_iter=40, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1).fit(Xtr, ytr)
best_svm = rand_svm.best_estimator_

def auc(clf): return roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
scores = {'LogReg': auc(best_lr), 'RandomForest': auc(best_rf), 'SVM_RBF': auc(best_svm)}
pd.Series(scores).to_csv(out/'tuned_auc_scores.csv')
joblib.dump(best_lr, 'models/tuned_logreg.joblib')
joblib.dump(best_rf, 'models/tuned_randomforest.joblib')
joblib.dump(best_svm, 'models/tuned_svm.joblib')
