Heart Disease Prediction (Cleveland)
A reproducible end‑to‑end machine learning project that predicts the presence of heart disease using the Cleveland subset of the UCI Heart Disease dataset. The repository includes data preparation, feature selection, hyperparameter tuning, a serialized pipeline, and an interactive Streamlit app for inference. For educational use only; not a clinical tool.

Key Features
Complete scikit‑learn pipeline: preprocessing + feature selection + final classifier.

Tuned models with persisted artifacts for instant inference.

Streamlit UI for interactive predictions.

Clear, step‑by‑step commands to reproduce results locally.

Project Structure

Heart_Disease_Project/
├── data/
│   └── heart_disease.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── models/
│   └── final_model.pkl
├── ui/
│   └── app.py
├── deployment/
│   └── ngrok_setup.txt
├── results/
│   └── evaluation_metrics.txt
├── README.md
├── requirements.txt
└── .gitignore


Optional extras (if present): tuned_*.joblib under models/, intermediate artifacts under results/ (e.g., selected_feature_indices.npy, tuned_auc_scores.csv), and utility scripts (e.g., scripts/refit_final_pipeline.py).

Getting Started
Prerequisites
Python 3.10+ (recommended: fresh virtual environment).

Install dependencies:

text
pip install -r requirements.txt
Reproducible Pipeline Build
From the project root, run:

text
python -m notebooks.01_data_preprocessing
python -m notebooks.03_feature_selection
python -m notebooks.06_hyperparameter_tuning
python -m models.build_final_pipeline
The last step saves a fitted pipeline to models/final_model.pkl and prints the selected best model.

Run the App
text
streamlit run ui/app.py
Open the local URL shown in the terminal and submit inputs to obtain predicted class and probability.

Usage Notes
Expected inputs: age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal.

Predicted class uses a 0.5 probability threshold by default. Adjust if a different operating point is required.

Troubleshooting
Pipeline is not fitted yet:

Rebuild using the steps in “Reproducible Pipeline Build.”

If needed, run a refit script (if provided) to retrain the full pipeline, then restart the app.

Custom transformer not found (during unpickling):

Ensure any custom classes (e.g., ColumnSlicer) are importable or defined at the top of ui/app.py before loading the pickle.

Caching issues:

Stop the app, clear cache, and relaunch.

Methodology Overview
Preprocessing: imputation (numeric/categorical), scaling numeric features, one‑hot encoding with unknown‑category handling.

Feature selection: statistical/importance‑based selection with persisted indices for consistent inference.

Model training and tuning: multiple classifiers evaluated; the best by AUC/ROC is persisted.

Persistence: final artifact is a scikit‑learn Pipeline including preprocessing, selection, and the tuned classifier.

Results
The build step logs the chosen best model (e.g., RandomForest) and stores models/final_model.pkl.

Consider adding results/evaluation_metrics.txt (accuracy, precision, recall, F1, AUC) and UI screenshots for completeness.

Dataset
UCI Heart Disease (Cleveland subset). Ensure compliance with dataset terms. Either include data/heart_disease.csv or document how it is obtained/generated.

Roadmap (Optional)
Unit tests for transformers and prediction interface.

CI workflow to lint, test, and package the app.

Optional cloud deployment (containerization + free PaaS).

License
Educational use only. Add a formal license (e.g., MIT) if redistribution or reuse is expected.

Acknowledgments
UCI Machine Learning Repository (Heart Disease dataset).

scikit‑learn and Streamlit communities.





