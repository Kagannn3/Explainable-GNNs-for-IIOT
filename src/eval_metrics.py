# src/eval_metrics.py

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

def extended_metrics(trues, preds):
    """
    Compute MAE, RMSE, R2, and normalized MAE.
    """
    mae  = mean_absolute_error(trues, preds)
    mse  = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(trues, preds)
    # Avoid division by zero
    span = max(trues) - min(trues)
    nmae = mae / span if span > 0 else np.nan
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'nMAE': nmae}

def alarm_metrics(trues, preds, Tth=30):
    """
    Threshold‐based alarm metrics: treat RUL ≤ Tth as 'about to fail'.
    Returns precision, recall, F1, false alarm rate (FAR), hit rate (HTR).
    """
    # Binarize
    y_true = [1 if t <= Tth else 0 for t in trues]
    y_pred = [1 if p <= Tth else 0 for p in preds]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    far       = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    htr       = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FAR': far,
        'HTR': htr,
    }
