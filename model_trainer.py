import os
# CRITICAL: Limit the parallelism of the underlying math libraries
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, max_error, median_absolute_error, 
    mean_absolute_percentage_error
)

def make_dense(x):
    return x.toarray()

def train_model(df):
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Train only on the 'text' column, ignore row_number to prevent overfitting on index
    X_train = train_df['text']
    y_train = train_df['score']
    X_val = val_df['text']
    y_val = val_df['score']
    
    # Optimized TF-IDF Vectorizer with stop words, bigrams, and more features
    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Dense transformer since BayesianRidge does not support sparse input matrices
    to_dense = FunctionTransformer(make_dense, accept_sparse=True)
    
    # BayesianRidge model to support predictive standard deviation / uncertainty metrics
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('to_dense', to_dense),
        ('regressor', BayesianRidge())
    ])

    model.fit(X_train, y_train)
    
    # Helper to calculate prediction metrics
    def get_metrics(X, y_true):
        X_trans = model.named_steps['vectorizer'].transform(X)
        X_dense = model.named_steps['to_dense'].transform(X_trans)
        y_pred, y_std = model.named_steps['regressor'].predict(X_dense, return_std=True)
        
        # Bound predictions to [0.0, 1.0] since it's a regression score
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        # Calculate confidence metric
        conf = 1.0 / (1.0 + y_std)
        
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

    # Calculate metrics for both sets
    train_results = get_metrics(X_train, y_train)
    val_results = get_metrics(X_val, y_val)

    metrics = {
        "message": "Model trained on 70% / Validated on 30% (Text only, BayesianRidge)",
        "data_split_info": {
            "rows_trained": len(X_train),
            "rows_validated": len(X_val)
        },
        "seen_data_metrics": train_results,
        "unseen_data_metrics": val_results
    }
    
    return model, metrics

def predict_with_confidence(model, texts, chunk_size=5000):
    """
    Given a model pipeline and a series of text inputs,
    return predictions and their calculated confidence.
    Processes in chunks to prevent memory allocation errors.
    """
    vectorizer = model.named_steps['vectorizer']
    to_dense = model.named_steps['to_dense']
    regressor = model.named_steps['regressor']
    
    # We will process in chunks of 5000 to save memory
    all_preds = []
    all_stds = []
    
    # Convert texts to a list if it's a pandas Series/Index
    texts_list = list(texts)
    
    for start_idx in range(0, len(texts_list), chunk_size):
        chunk = texts_list[start_idx : start_idx + chunk_size]
        
        # Transform chunk
        X_trans = vectorizer.transform(chunk)
        X_dense = to_dense.transform(X_trans)
        
        # Predict mean and standard deviation (uncertainty) for this chunk
        y_pred_chunk, y_std_chunk = regressor.predict(X_dense, return_std=True)
        
        all_preds.append(y_pred_chunk)
        all_stds.append(y_std_chunk)
        
    y_pred = np.concatenate(all_preds)
    y_std = np.concatenate(all_stds)
    
    # Bound predictions
    y_pred = np.clip(y_pred, 0.0, 1.0)
    
    # Compute confidence: 1 / (1 + standard_deviation)
    confidence = 1.0 / (1.0 + y_std)
    
    return y_pred, confidence
