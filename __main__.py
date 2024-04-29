import scipy.io, torch

import pandas as pd, numpy as np

data = scipy.io.loadmat("cylinder_nektar_wake.mat")

U_star = data['U_star'] # N x 2 x T
p_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

# Averaging U, V and p over the time dimension
U_mean = U_star.mean(axis=2)[:, 0]  # Averaging over time, selecting U component
V_mean = U_star.mean(axis=2)[:, 1]  # Averaging over time, selecting V component
p_mean = p_star.mean(axis=1)  # Average pressure over time

# Create a DataFrame with averaged values
df_raissi = pd.DataFrame({
    'x': X_star[:, 0],
    'y': X_star[:, 1],
    'u': U_mean,
    'v': V_mean,
    'p': p_mean
})

def add_cylinder_points(df, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = 0.5 * np.cos(angles)
    y = 0.5 * np.sin(angles)
    df_new = pd.DataFrame({
        'x': x,
        'y': y,
        'u': 0,
        'v': 0,
        'p': np.nan
    })
    return pd.concat([df_new, df], ignore_index=True)

df_raissi_cyl = add_cylinder_points(df_raissi, num_points=200)

def expand_dataframe_with_boundary_points(df, int_num_points, ext_num_points):
    # Generate random points within the specified range
    x = np.random.uniform(-15, 8, int_num_points)
    y = np.random.uniform(-8, 8, int_num_points)
    
    # Coordinates close to the cylinder but outside it
    r = 0.5 + np.random.uniform(0.001, 1, int_num_points)  # Slightly outside the radius of 0.5
    angles = np.random.uniform(0, 2 * np.pi, int_num_points)
    
    # Points close to the cylinder
    x_close = r * np.cos(angles)
    y_close = r * np.sin(angles)
    
        # Define the bounds of the rectangle
    x_min, x_max, y_min, y_max = -15, 8, -8, 8
    
    # Evenly spaced points along each side
    x_left = np.full(ext_num_points, x_min)
    x_right = np.full(ext_num_points, x_max)
    y_bottom = np.full(ext_num_points, y_min)
    y_top = np.full(ext_num_points, y_max)
    
    y_left = np.linspace(y_min, y_max, ext_num_points)
    y_right = np.linspace(y_min, y_max, ext_num_points)
    x_bottom = np.linspace(x_min, x_max, ext_num_points)
    x_top = np.linspace(x_min, x_max, ext_num_points)

    # Combine all x and y, including their symmetric counterparts
    all_x = np.concatenate([x, x, x_close, x_close, x_left, x_right, x_bottom, x_top])
    all_y = np.concatenate([y, -y, y_close, -y_close, y_left, y_right, y_bottom, y_top])
    
    # DataFrame of new points
    df_new = pd.DataFrame({
        'x': all_x,
        'y': all_y
    })

    # Removing duplicates by considering only unique rows
    df_new = df_new.drop_duplicates().reset_index(drop=True)

    # Add NaNs for the other columns
    df_new['u'] = np.nan
    df_new['v'] = np.nan
    df_new['p'] = np.nan

    # Append the new DataFrame to the original, avoiding duplicates in (x, y)
    df_combined = pd.concat([df, df_new])
    df_combined = df_combined.drop_duplicates(subset=['x', 'y']).reset_index(drop=True)
    
    # Set u = 1 and v = 0 where x = -15
    df_combined.loc[df_combined['x'] == x_min, 'u'] = 1
    df_combined.loc[df_combined['x'] == x_min, 'v'] = 0

    return df_combined

df_train = expand_dataframe_with_boundary_points(df_raissi_cyl, 5000, 200)

def add_sdf_column(df):
    # Compute the radius from the center of the cylinder
    r = 0.5
    # Calculate the distance of each point (x, y) from the center (0,0)
    distances = np.sqrt(df['x']**2 + df['y']**2)
    # Calculate signed distance function (sdf)
    df['sdf'] = distances - r
    return df

df_train = add_sdf_column(df_train)

# Create a mask for non-negative y values
mask = df_train['y'] > 0

df_sym_pos = df_train[mask]
df_sym_neg = df_train[~mask]

df_sym_pos = df_sym_pos.copy()
df_sym_neg = df_sym_neg.copy()

df_sym_pos.sort_values(by=['x', 'y'], inplace=True)
df_sym_neg.sort_values(by=['x', 'y'], ascending=[True, False], inplace=True)

df_train = pd.concat([df_sym_pos, df_sym_neg], ignore_index=True)

l = len(df_sym_pos)
sym_mask = df_train.index >= l

bc_mask = (df_train['sdf'] < 1e-15) & (df_train['sdf'] > -1e-15)

raissi_mask = df_train.notna().all(axis=1)

print("Dataset Loaded.")

import torch
# sys.path.append('/Users/leondeligny/Desktop/PDM/PINN/Raissi')

from PINNsFormer import PINNsFormer
# from FourierNeuralOperatorMNN import FourierNeuralOperatorMNN

# Train the model
model = PINNsFormer(df_train, bc_mask, sym_mask, raissi_mask)
# model = PINN_Raissi(df_train, bc_mask, sym_mask)
# model = FourierNeuralOperatorMNN(tensor_data, bc_indices, sym_pos_indices, sym_neg_indices)
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
u_pred, v_pred, p_pred, f_u_pred, f_v_pred, ic_pred = model.predict()

f_u_pred_tensor = torch.tensor(f_u_pred, dtype=torch.float32)
f_v_pred_tensor = torch.tensor(f_v_pred, dtype=torch.float32)
ic_pred_tensor = torch.tensor(ic_pred, dtype=torch.float32)

# Plotting
plot_predictions_vs_test(df_raissi['x'].astype(float).values.flatten(), df_raissi['y'].astype(float).values.flatten(), u_pred, df_raissi['u'], 'u', 'PINN_Raissi_FNO')
plot_predictions_vs_test(df_raissi['x'].astype(float).values.flatten(), df_raissi['y'].astype(float).values.flatten(), v_pred, df_raissi['v'], 'v', 'PINN_Raissi_FNO')
plot_predictions_vs_test(df_raissi['x'].astype(float).values.flatten(), df_raissi['y'].astype(float).values.flatten(), p_pred, df_raissi['p'], 'p', 'PINN_Raissi_FNO')

plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), f_u_pred_tensor, torch.zeros_like(f_u_pred_tensor), 'f_u', 'PINN_Raissi_FNO')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), f_v_pred_tensor, torch.zeros_like(f_v_pred_tensor), 'f_v', 'PINN_Raissi_FNO')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), ic_pred_tensor, torch.zeros_like(ic_pred_tensor), 'ic', 'PINN_Raissi_FNO')