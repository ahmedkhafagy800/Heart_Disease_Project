# ui/app.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSlicer(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = np.array(indices)
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices]

import streamlit as st, pandas as pd, joblib
from pathlib import Path
from sklearn.utils.validation import check_is_fitted  # fitted check

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Prediction (Cleveland)")

model_path = Path('models/final_model.pkl')
if not model_path.exists():
    st.error("Run training and build steps to create models/final_model.pkl"); st.stop()

pipe = joblib.load(model_path)

# تحقق أن البايبلاين/الموديل متدرّب
try:
    check_is_fitted(pipe)
except Exception:
    try:
        check_is_fitted(pipe.named_steps.get('model', pipe))
    except Exception as e:
        st.error(f"Loaded pipeline is not fitted: {e}. Rebuild it then restart the app."); st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('age', 18, 100, 54)
    trestbps = st.number_input('trestbps (mm Hg)', 80, 220, 130)
    chol = st.number_input('chol (mg/dl)', 100, 600, 246)
    thalach = st.number_input('thalach', 60, 220, 150)
with col2:
    oldpeak = st.number_input('oldpeak', 0.0, 10.0, 1.0, step=0.1)
    sex = st.selectbox('sex (1=male,0=female)', [0,1], 1)
    cp = st.selectbox('cp', [0,1,2,3], 3)
    fbs = st.selectbox('fbs', [0,1], 0)
    restecg = st.selectbox('restecg', [0,1,2], 1)
    exang = st.selectbox('exang', [0,1], 0)
    slope = st.selectbox('slope', [0,1,2], 1)
    ca = st.selectbox('ca', [0,1,2,3], 0)
    thal = st.selectbox('thal', [0,1,2,3,6,7], 5)

def get_probability(clf, Xdf):
    if hasattr(clf, "predict_proba"):
        return float(clf.predict_proba(Xdf)[0, 1])
    elif hasattr(clf, "decision_function"):
        # تحويل decision_function لاحتمال تقريبي
        z = float(clf.decision_function(Xdf))
        return 1 / (1 + np.exp(-z))
    else:
        # fallback من التنبؤ مباشرة
        return float(clf.predict(Xdf))

if st.button("Predict"):
    X = pd.DataFrame([{'age':age,'trestbps':trestbps,'chol':chol,'thalach':thalach,'oldpeak':oldpeak,
                       'sex':sex,'cp':cp,'fbs':fbs,'restecg':restecg,'exang':exang,'slope':slope,'ca':ca,'thal':thal}])
    prob = get_probability(pipe, X)
    pred = int(prob >= 0.5)
    st.metric("Predicted class", f"{pred}")
    st.metric("Probability of presence", f"{prob:.3f}")
    st.caption("For educational use only.")
