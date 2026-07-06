import os
# CRITICAL: Limit the parallelism of the underlying math libraries
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # <-- Added back
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, max_error, median_absolute_error, 
    mean_absolute_percentage_error
)
def train_model(df):
    
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    
    X_train = train_df[['row_number', 'text']]
    y_train = train_df['score']
    X_val = val_df[['row_number', 'text']]
    y_val = val_df['score']
    
    # Preprocessor and Model setup
    preprocessor = ColumnTransformer(
        transformers=[('text', TfidfVectorizer(max_features=500), 'text')],
        remainder='passthrough',
        sparse_threshold=0
    )
    model = Pipeline([('preprocessor', preprocessor), ('regressor', BayesianRidge())])

    model.fit(X_train, y_train)
    
    # Helper to get predictions and confidence
    def get_metrics_and_conf(X, y_true):
        X_trans = model.named_steps['preprocessor'].transform(X)
        y_pred, y_std = model.named_steps['regressor'].predict(X_trans, return_std=True)
        conf = 100 * (1 / (1 + y_std))
        
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "medae": float(median_absolute_error(y_true, y_pred)),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
            "max_error": float(max_error(y_true, y_pred)),
            "confidence": {
                "avg": float(np.mean(conf)),
                "min": float(np.min(conf)),
                "max": float(np.max(conf))
            }
        }

    # 3. Calculate metrics for both sets
    train_results = get_metrics_and_conf(X_train, y_train)
    val_results = get_metrics_and_conf(X_val, y_val)

    return ({
        "message": "Model trained on 70% / Validated on 30%",
        "data_split_info": {
            "rows_trained": len(X_train),
            "rows_validated": len(X_val)
        },
        "seen_data_metrics": train_results,
        "unseen_data_metrics": val_results
    })

