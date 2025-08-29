# data/load_data.py
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_uci_heart():
    hd = fetch_ucirepo(id=45)
    X = hd.data.features.copy()
    y = hd.data.targets.copy()
    target_col = 'num' if 'num' in y.columns else y.columns
    y_bin = (y[target_col].astype(float) > 0).astype(int).rename('target')
    df = pd.concat([X, y_bin], axis=1)

    wanted = ['age','sex','cp','trestbps','chol','fbs','restecg',
              'thalach','exang','oldpeak','slope','ca','thal']
    # صنع الأعمدة إن كانت أسماؤها بديلة
    alias = {'thalachh':'thalach'}
    for k,v in alias.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # تحويل الأنواع
    for c in wanted:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['age'])
    X = df[wanted].copy()
    y = df['target'].copy()
    return X, y
