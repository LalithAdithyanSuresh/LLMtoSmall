import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

def evaluate_predictions(json_path, csv_path):
    print(f"Loading predictions from: {json_path}")
    print(f"Loading true labels from: {csv_path}")
    
    # 1. Load predictions JSON
    if not os.path.exists(json_path):
        print(f"Error: Predictions file not found at {json_path}")
        return
        
    with open(json_path, "r") as f:
        task_data = json.load(f)
        
    final_preds = task_data.get("final_predictions", [])
    if not final_preds:
        print("Error: No 'final_predictions' key or empty predictions found in task JSON.")
        return
        
    df_preds = pd.DataFrame(final_preds)
    
    # 2. Load and preprocess original CSV (replicate the exact cleaning pipeline)
    if not os.path.exists(csv_path):
        print(f"Error: Original CSV not found at {csv_path}")
        return
        
    df_orig = pd.read_csv(csv_path)
    
    # Apply exact clean: dropna on text, reset index, rename index to row_number
    df_orig = df_orig.dropna(subset=['text']).reset_index().rename(columns={'index': 'row_number'})
    
    # 3. Merge predictions and true labels on row_number
    # True label is in 'spam' column, predicted score in 'score'
    df_eval = pd.merge(df_orig[['row_number', 'spam', 'text']], df_preds, on='row_number')
    
    print(f"Successfully matched {len(df_eval)} predictions with true labels.")
    
    y_true = df_eval['spam'].values
    y_score = df_eval['score'].values
    y_pred = (y_score >= 0.5).astype(int)
    y_conf = df_eval['confidence'].values
    
    # 4. Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("CLASSIFICATION PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("="*50)
    print(f"Confusion Matrix:\n{cm}")
    print("="*50)
    
    # 5. Generate beautiful plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Student Model Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Confusion Matrix
    ax = axes[0, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)
    classes = ['Ham (0)', 'Spam (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.grid(False)
    
    # Plot 2: ROC Curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    
    # Plot 3: Prediction Score Distribution (Spam vs Ham)
    ax = axes[1, 0]
    ax.hist(y_score[y_true == 0], bins=25, alpha=0.6, color='dodgerblue', label='True Ham (0)', edgecolor='black')
    ax.hist(y_score[y_true == 1], bins=25, alpha=0.6, color='crimson', label='True Spam (1)', edgecolor='black')
    ax.set_xlabel('Predicted Spam Probability Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc="upper center", fontsize=11)
    
    # Plot 4: Accuracy/Error Rate by Confidence Quantiles
    ax = axes[1, 1]
    # Group predictions into bins of confidence
    df_eval['correct'] = (df_eval['spam'] == y_pred).astype(int)
    # Bin by confidence
    df_eval['conf_bin'] = pd.qcut(df_eval['confidence'], q=5, duplicates='drop')
    bin_stats = df_eval.groupby('conf_bin', observed=False).agg(
        avg_conf=('confidence', 'mean'),
        accuracy=('correct', 'mean'),
        count=('correct', 'count')
    ).reset_index()
    
    ax.plot(bin_stats['avg_conf'], bin_stats['accuracy'], marker='o', color='forestgreen', lw=3, markersize=8, label='Model Accuracy')
    ax.plot(bin_stats['avg_conf'], bin_stats['avg_conf'], color='grey', linestyle='--', lw=2, label='Perfect Calibration')
    ax.set_xlabel('Average Prediction Confidence', fontsize=12)
    ax.set_ylabel('Empirical Accuracy', fontsize=12)
    ax.set_title('Confidence Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plot_save_path = "temp/evaluation_results.png"
    plt.savefig(plot_save_path, dpi=150)
    print(f"\nEvaluation plots successfully generated and saved to: {plot_save_path}")
    
    # Write a quick markdown summary file
    markdown_path = "temp/evaluation_report.md"
    with open(markdown_path, "w") as f:
        f.write(f"""# Student Model Evaluation Report

This report evaluates the accuracy of the student regression model (`BayesianRidge`) trained on LLM annotations against the true labels from the original dataset.

## Core Classification Performance

- **Matched Row Count**: {len(df_eval)}
- **Accuracy**: {accuracy:.4%}
- **Precision**: {precision:.4%}
- **Recall**: {recall:.4%}
- **F1-Score**: {f1:.4%}
- **ROC AUC Score**: {roc_auc:.4f}

### Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **True Ham (0)** | {cm[0, 0]} | {cm[0, 1]} |
| **True Spam (1)** | {cm[1, 0]} | {cm[1, 1]} |

## Visualizations
The evaluation charts have been generated and saved to:
`{os.path.abspath(plot_save_path)}`

*Includes Confusion Matrix, ROC Curve, Prediction Probability Distribution, and Confidence Calibration curves.*
""")
    print(f"Evaluation report written to: {markdown_path}")

if __name__ == "__main__":
    evaluate_predictions(
        "tasks/1_Extract-spam-email-text.json",
        "temp/emails_shuffled.csv"
    )
