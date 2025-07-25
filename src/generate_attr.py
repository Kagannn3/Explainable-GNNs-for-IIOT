import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients
import os

# ===========================
# Model Definition (Same as train_client.py)
# ===========================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use output of last time step
        return out


# ===========================
# Load Model
# ===========================

model_path = "models/client0_model.pt"
model = LSTMModel(input_dim=4, hidden_dim=8, output_dim=2)  # ← adjust if needed
model.load_state_dict(torch.load(model_path))
model.eval()


# ===========================
# Prepare Sample Input
# ===========================

# Sample input (seq_len=3, input_dim=4)
sample_input = torch.tensor([[
    [0.1, 0.3, 0.2, 0.5],
    [0.0, 0.1, 0.4, 0.3],
    [0.2, 0.2, 0.1, 0.6]
]], dtype=torch.float32, requires_grad=True)

# Choose the target class index
target_class = 1


# ===========================
# Attribution using Captum
# ===========================

ig = IntegratedGradients(model)
attr, delta = ig.attribute(sample_input, target=target_class, return_convergence_delta=True)

print("Attribution shape:", attr.shape)  # e.g. [1, 3, 4]
print("Attributions:\n", attr)


# ===========================
# Save to .npy File
# ===========================

output_dir = "explanations"
os.makedirs(output_dir, exist_ok=True)
attr_numpy = attr.detach().numpy()
np.save(os.path.join(output_dir, "client0_lstm_attr.npy"), attr_numpy)

print("Attribution saved to:", os.path.join(output_dir, "client0_lstm_attr.npy"))
