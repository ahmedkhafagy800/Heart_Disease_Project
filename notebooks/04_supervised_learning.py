import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

out = Path('results'); out.mkdir(exist_ok=True, parents=True)
Xtr = np.load(out/'X_train_preprocessed.npy'); Xte = np.load(out/'X_test_preprocessed.npy')
ytr = pd.read_csv(out/'y_train.csv').iloc[:,0]; yte = pd.read_csv(out/'y_test.csv').iloc[:,0]
sel = np.load(out/'selected_feature_indices.npy')
Xtr = Xtr[:, sel]; Xte = Xte[:, sel]

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, solver='liblinear'),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=400, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42)
}

rows = []
for name, clf in models.items():
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:,1] if hasattr(clf,'predict_proba') else None
    acc = accuracy_score(yte, yp); prec = precision_score(yte, yp, zero_division=0)
    rec = recall_score(yte, yp, zero_division=0); f1 = f1_score(yte, yp, zero_division=0)
    auc = roc_auc_score(yte, yprob) if yprob is not None else float('nan')
    rows.append([name, acc, prec, rec, f1, auc])

    disp = ConfusionMatrixDisplay.from_predictions(yte, yp)
    disp.figure_.tight_layout(); disp.figure_.savefig(out/f'cm_{name}.png'); plt.close(disp.figure_)
    if yprob is not None:
        roc = RocCurveDisplay.from_predictions(yte, yprob)
        roc.figure_.tight_layout(); roc.figure_.savefig(out/f'roc_{name}.png'); plt.close(roc.figure_)

pd.DataFrame(rows, columns=['model','accuracy','precision','recall','f1','auc']).to_csv(out/'supervised_metrics.csv', index=False)

for name, clf in models.items():
    joblib.dump(clf, f'models/base_{name}.joblib')
