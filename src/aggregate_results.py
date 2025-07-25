import json, pandas as pd
# 📊 Add this part directly below
import matplotlib.pyplot as plt

records = []
for cid in [0, 1, 2]:
    with open(f'results/metrics_client{cid}.json') as f:
        m = json.load(f)
    for model in ['lstm', 'gcn', 'gat']:
        records.append({
            'Client': cid,
            'Model': model.upper(),
            'MAE': m[model]['mae'],
            'RMSE': m[model]['rmse']
        })

df = pd.DataFrame(records)
avg_df = df.groupby('Model').mean()
print(avg_df)

# Plotting the average metrics
data = {
    'Model': ['LSTM', 'GCN', 'GAT'],
    'MAE': avg_df['MAE'].tolist(),
    'RMSE': avg_df['RMSE'].tolist()
}

plot_df = pd.DataFrame(data)

plot_df.plot(x='Model', kind='bar', figsize=(8, 5), rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Error')
plt.grid(axis='y')
plt.tight_layout()
plt.show()