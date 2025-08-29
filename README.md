Heart Disease Prediction (Cleveland)
A reproducible end‑to‑end machine learning project that predicts the presence of heart disease using the Cleveland subset of the UCI Heart Disease dataset. The repository includes data preparation, feature selection, hyperparameter tuning, a serialized pipeline, and an interactive Streamlit app for inference. For educational use only; not a clinical tool.

Key Features
Complete scikit‑learn pipeline: preprocessing + feature selection + final classifier.

Tuned models with persisted artifacts for instant inference.

Streamlit UI for interactive predictions.

Clear, step‑by‑step commands to reproduce results locally.

Project Structure
data/: source dataset (heart_disease.csv).

notebooks/: scripts/notebooks for preprocessing, PCA/selection, supervised training, and tuning.

results/: intermediate artifacts (e.g., selected_feature_indices.npy, tuned_auc_scores.csv).

models/: final_model.pkl (fitted pipeline) and tuned_*.joblib artifacts.

ui/: Streamlit app (app.py).

scripts/: utility scripts (e.g., refit_final_pipeline.py for emergency refit).

README.md, requirements.txt, .gitignore.

Getting Started
Prerequisites
Python 3.10+ (recommend a fresh virtual environment).

Install dependencies:

pip install -r requirements.txt

Reproducible Pipeline Build
From the project root, run:

python -m notebooks.01_data_preprocessing

python -m notebooks.03_feature_selection

python -m notebooks.06_hyperparameter_tuning

python -m models.build_final_pipeline

The last step saves a fitted pipeline to models/final_model.pkl and prints the selected best model.

Run the App
streamlit run ui/app.py

Open the local URL shown in the terminal and use the form to submit inputs and obtain a predicted class and probability.

Usage Notes
The app expects the following inputs: age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal.

Predicted class is determined by a 0.5 probability threshold; adjust logic if a different operating point is desired.

Troubleshooting
Pipeline is not fitted yet:

Rebuild artifacts using the steps in “Reproducible Pipeline Build.”

If needed, run scripts/refit_final_pipeline.py to refit the entire pipeline on raw features, then restart the app.

Custom transformer not found (unpickling issues):

Ensure the ColumnSlicer class is defined at the top of ui/app.py before loading the pickle (or import it from a shared module that’s resolvable at runtime).

Caching issues:

Stop the app and clear cache, then relaunch.

Methodology Overview
Preprocessing: imputation (numeric/ categorical), scaling for numeric features, and one‑hot encoding with unknown‑category handling.

Feature selection: statistical/importance‑based selection with persisted indices for consistent inference.

Model training and tuning: multiple classifiers evaluated; the best by AUC/ROC is selected and persisted.

Persistence: the final artifact is a scikit‑learn Pipeline containing preprocessing, selection, and the tuned classifier.

Results
The build step prints the chosen best model (e.g., RandomForest) and stores the final pipeline as models/final_model.pkl.

Consider adding results/evaluation_metrics.txt (accuracy, precision, recall, F1, AUC) and screenshots of the Streamlit UI for completeness.

Dataset
UCI Heart Disease (Cleveland subset). Ensure compliance with the dataset’s usage terms. Include the CSV under data/ or document the acquisition process.

Roadmap (Optional)
Add unit tests for data transformers and prediction interface.

CI workflow to lint, test, and package the app.

Optional cloud deployment (e.g., containerize and deploy to a free PaaS).

License
Educational use only. Add a formal license (e.g., MIT) if redistribution or reuse is expected.

Acknowledgments
UCI Machine Learning Repository (Heart Disease dataset).

scikit‑learn and Streamlit communities for tooling and best practices.


# Project’s heart disease dataset and loaders
├── data/ 
│ ├── heart_disease.csv # Optional; may be omitted to reduce repo size
│ └── load_data.py # Script to download/load data if CSV omitted
│
├── notebooks/ # One script/notebook per step
│ ├── 01_data_preprocessing.py (or .ipynb)
│ ├── 02_pca_analysis.py (or .ipynb)
│ ├── 03_feature_selection.py (or .ipynb)
│ ├── 04_supervised_learning.py (or .ipynb)
│ ├── 05_unsupervised_learning.py (or .ipynb)
│ └── 06_hyperparameter_tuning.py (or .ipynb)
│
├── models/
│ ├── build_final_pipeline.py
│ ├── final_model.pkl # Fitted pipeline artifact
│ └── tuned_*.joblib # Any tuned model artifacts
│
├── results/ # All plots and evaluation outputs
│ ├── figures/
│ │ ├── histograms/
│ │ ├── heatmap/
│ │ ├── boxplots/
│ │ ├── pca_plots/
│ │ ├── feature_selection/
│ │ ├── roc_curves/
│ │ └── confusion_matrices/
│ ├── supervised_metrics.csv
│ └── clustering_metrics.txt
│
├── ui/
│ └── app.py # Streamlit interface
│
├── deployment/
│ └── ngrok_setup.txt # Optional local sharing notes
│
├── scripts/ # Optional utilities (e.g., refit script)
│ └── refit_final_pipeline.py
│
├── README.md
├── requirements.txt
└── .gitignore
