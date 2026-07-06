# Student Model Evaluation Report

This report evaluates the accuracy of the student regression model (`BayesianRidge`) trained on LLM annotations against the true labels from the original dataset.

## Core Classification Performance

- **Matched Row Count**: 5728
- **Accuracy**: 95.9672%
- **Precision**: 93.4966%
- **Recall**: 89.3275%
- **F1-Score**: 91.3645%
- **ROC AUC Score**: 0.9892

### Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **True Ham (0)** | 4275 | 85 |
| **True Spam (1)** | 146 | 1222 |

## Visualizations
The evaluation charts have been generated and saved to:
`D:\Internship\LLMtoSmall\temp\evaluation_results.png`

*Includes Confusion Matrix, ROC Curve, Prediction Probability Distribution, and Confidence Calibration curves.*
