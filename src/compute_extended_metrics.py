#!/usr/bin/env python3
"""
compute_extended_metrics.py

Loads per-client raw predictions & ground truths, computes:
  • Regression metrics: MAE, RMSE, R², nMAE
  • Alarm metrics: Precision, Recall, F1, False Alarm Rate, Hit Rate
Saves results to results/extended_metrics.csv
"""

import os
import numpy as np
import pandas as pd

from eval_metrics import extended_metrics, alarm_metrics

def main():
    os.makedirs('results', exist_ok=True)

    records = []
    for cid in [0, 1, 2]:
        for model in ['LSTM', 'GCN', 'GAT']:
            trues_path = f"results/trues_client{cid}_{model}.npy"
            preds_path = f"results/preds_client{cid}_{model}.npy"
            if not (os.path.exists(trues_path) and os.path.exists(preds_path)):
                print(f"[WARN] Missing files for client {cid}, model {model}")
                continue

            # Load arrays
            trues = np.load(trues_path)
            preds = np.load(preds_path)

            # Compute metrics
            em = extended_metrics(trues, preds)
            am = alarm_metrics(trues, preds, Tth=30)

            # Flatten into one record
            rec = {
                'Client': cid,
                'Model': model,
                **em,
                **am
            }
            records.append(rec)

    # Build DataFrame
    df = pd.DataFrame(records)
    df.to_csv('results/extended_metrics.csv', index=False)
    print(">> Saved extended metrics to results/extended_metrics.csv\n")

    # Print average per model
    avg = df.groupby('Model').mean()[['MAE','RMSE','R2','nMAE','Precision','Recall','F1','FAR','HTR']]
    print("Average metrics by model:\n", avg)

if __name__ == '__main__':
    main()
