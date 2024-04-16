!/usr/bin/env python3

import scipy.io

import pandas as pd

data = scipy.io.loadmat("/Users/leondeligny/Desktop/PDM/PINN/Raissi/cylinder_nektar_wake.mat")

U_star = data['U_star'] # N x 2 x T
p_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

# Averaging U, V and p over the time dimension
U_mean = U_star.mean(axis=2)[:, 0]  # Averaging over time, selecting U component
V_mean = U_star.mean(axis=2)[:, 1]  # Averaging over time, selecting V component
p_mean = p_star.mean(axis=1)  # Average pressure over time

# Create a DataFrame with averaged values
df_train = pd.DataFrame({
    'x': X_star[:, 0],
    'y': X_star[:, 1],
    'u': U_mean,
    'v': V_mean,
    'p': p_mean
})

bc_mask = (df_train['x'] == df_train['x'].min()) | \
       (df_train['x'] == df_train['x'].max()) | \
       (df_train['y'] == df_train['y'].min()) | \
       (df_train['y'] == df_train['y'].max())

df_bc = df_train[bc_mask]

print(df_bc)

import torch, sys
# sys.path.append('/Users/leondeligny/Desktop/PDM/PINN/Raissi')

from PINN_Raissi import PINN_Raissi

mean_variance_dict = {}

# Train the model
model = PINN_Raissi(df_train, bc_mask)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
    
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Started Training.")
model.train(501)
print(f"Finished Training.")

from plot import plot_predictions_vs_test

# Prediction u_pred, v_pred, p_pred, nut_pred
u_pred, v_pred, p_pred = model.predict(df_train)

# Plotting
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), u_pred, df_train['u'], 'u', 'PINN_Raissi_FourierF')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), v_pred, df_train['v'], 'v', 'PINN_Raissi_FourierF')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), p_pred, df_train['p'], 'p', 'PINN_Raissi_FourierF')
