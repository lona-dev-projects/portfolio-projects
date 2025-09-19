import pandas as pd
import numpy as np
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

@dataclass
class Config:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    INPUT_FILEPATH: Path = _PROJECT_ROOT / 'data' / 'processed' / 'eurusd_m5_ml_ready.parquet'
    MODEL_OUTPUT_PATH: Path = _PROJECT_ROOT / 'models' / 'eurusd_lgbm_baseline.pkl'
    SHAP_PLOT_PATH: Path = _PROJECT_ROOT / 'reports' / 'eurusd_feature_importance_shap.png'

    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.15 

    LGBM_PARAMS: dict = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'max_depth': 8,
        'num_leaves': 63,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })
    EARLY_STOPPING_ROUNDS: int = 50

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the feature-rich dataset."""
    print(f"--> Loading data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        print(f"✅ Loaded {len(df):,} data points with {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input data file not found at {filepath}")
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    print("--> Preparing data for training...")
    X = df.drop(columns=['target'])
    y = df['target']
    
    target_mapping = {-1: 0, 0: 1, 1: 2}
    y = y.map(target_mapping)
    print("✅ Target remapped: -1→0, 0→1, 1→2")
    return X, y
def split_data(X: pd.DataFrame, y: pd.Series, config: Config) -> tuple:
    """Performs a robust chronological train/validation/test split."""
    print(f"--> Creating chronological train/test split...")
    
    test_split_index = int(len(X) * (1 - config.TEST_SIZE))
    X_train_full, X_test = X[:test_split_index], X[test_split_index:]
    y_train_full, y_test = y[:test_split_index], y[test_split_index:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=config.VALIDATION_SIZE, shuffle=False
    )

    print(f"Training set size:   {len(X_train):,}")
    print(f"Validation set size: {len(X_val):,}")
    print(f"Test set size:       {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val, config: Config) -> lgb.LGBMClassifier:
    """Initializes and trains the LightGBM model with early stopping."""
    print("\n--- 🧠 Training Baseline LightGBM Model ---")
    model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True)]
    )
    print("✅ Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    print("\n--- 📊 Model Performance on Out-of-Sample Test Set ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    class_names = ['Stop Loss (0)', 'Time Limit (1)', 'Profit Take (2)']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}\n")

def analyze_feature_importance(model, X_train, file_path: Path):
    print("\n--- 🔬 Analyzing Feature Importance with SHAP ---")
    print("--> Calculating SHAP values (this may take a moment)...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, class_names=['SL', 'Timeout', 'PT'])
    plt.title("Mean SHAP Value (Feature Importance) for EUR/USD Model")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    print(f"✅ SHAP feature importance plot saved to: {file_path}")
    abs_shap_values = [np.abs(sv) for sv in shap_values]
    sum_abs_shap_across_classes = np.sum(abs_shap_values, axis=0)
    final_feature_importance = np.mean(sum_abs_shap_across_classes, axis=0)
    
    shap_df = pd.DataFrame({
        'SHAP_Importance': final_feature_importance
    }, index=X_train.columns)

    shap_df = shap_df.sort_values('SHAP_Importance', ascending=False)
    
    print("\nTop 15 Most Predictive Features:")
    print(shap_df.head(15).to_string())

def save_model(model, filepath: Path):
    """Saves the trained model to disk."""
    print(f"\n--> Saving model to {filepath}...")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print("✅ Model saved successfully.")

def main():
    start_time = time.time()
    cfg = Config()
    
    df = load_data(cfg.INPUT_FILEPATH)
    X, y = prepare_data(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, cfg)
    
    model = train_model(X_train, y_train, X_val, y_val, cfg)
    evaluate_model(model, X_test, y_test)
    analyze_feature_importance(model, X_train, cfg.SHAP_PLOT_PATH)
    save_model(model, cfg.MODEL_OUTPUT_PATH)
    
    end_time = time.time()
    print(f"\n--- Pipeline Complete in {end_time - start_time:.2f} seconds ---")
if __name__ == "__main__":
    main()
