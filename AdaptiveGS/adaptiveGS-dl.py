import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler 
import time
import psutil
import os
import joblib
from memory_profiler import memory_usage
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
import gc 

# --------------------------- Global Configuration ---------------------------
# Dataset paths
dataset_paths = {
    "gy_geno": "datasets/Wheat/gy_geno.csv",
    "gy_pheno": "datasets/Wheat/gy_pheno.csv"
}

# Phenotype dictionary
pheno_dict = {
    'gy_pheno': ['yield']
}

# Original hyperparameters
ORIGINAL_HYPERPARAMS = {
    'CNN': {'num_conv_layers': 3, 'dropout': 0.5, 'lr': 1e-3, 'num_epochs': 150, 'batch_size': 64, 'initial_out_channels': 32},
    'MLP': {'num_hidden_layers': 2, 'dropout': 0.5, 'lr': 1e-3, 'num_epochs': 150, 'batch_size': 64},
    'Transformer': {'num_layers': 2, 'dropout': 0.1, 'lr': 1e-3, 'num_epochs': 150, 'batch_size': 64, 'd_model': 64, 'nhead': 4}
}

# --------------------------- Data Preprocessing ---------------------------
def get_data(dataset_id_t, y_temp_col):
    """
    Safely loads, cleans, and standardizes data.
    Only phenotype data is standardized.
    Returns train/test sets as NumPy arrays and the fitted y scaler.
    """
    try:
        # Read data
        geno_path = dataset_paths[dataset_id_t + '_geno']
        pheno_path = dataset_paths[dataset_id_t + '_pheno']
        geno_data = pd.read_csv(geno_path)
        pheno_data = pd.read_csv(pheno_path)
        # geno_data = pd.read_csv(geno_path,nrows=20,usecols=range(10))
        # pheno_data = pd.read_csv(pheno_path,nrows=20)
        # Make sure the first column is the sample identifier in both files
        concat_data = pd.merge(
            pheno_data,
            geno_data,
            left_on=pheno_data.columns[0],
            right_on=geno_data.columns[0]
        )

        # Get feature columns (assuming first column in geno_data is ID)
        x_col = geno_data.columns.tolist()[1:]

        # Final data cleaning: keep only target and feature columns, drop rows with NaNs
        concat_data = concat_data[[y_temp_col] + x_col].dropna()

        if concat_data.empty:
             raise ValueError(f"After merging and dropping NaNs, no valid data remains for {dataset_id_t}_{y_temp_col}")

        # Separate features (X) and target (y)
        X = concat_data[x_col]
        y = concat_data[y_temp_col]

        # Split data into train and test sets
        X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        # Data Standardization
        # Only standardize y (phenotype)
        scaler_y = StandardScaler()
        # fit_transform expects 2D array, .values converts Series to NumPy array, reshape ensures 2D
        y_train_scaled = scaler_y.fit_transform(y_train_s.values.reshape(-1, 1)).flatten()
        # Use the scaler fitted on training data to transform test data
        y_test_scaled = scaler_y.transform(y_test_s.values.reshape(-1, 1)).flatten()

        X_train_np = X_train_df.values
        X_test_np = X_test_df.values

        print(f"Data loaded, cleaned, and phenotype standardized. X_train shape: {X_train_np.shape}, y_train_scaled shape: {y_train_scaled.shape}")
        return X_train_np, X_test_np, y_train_scaled, y_test_scaled, scaler_y

    except Exception as e:
        raise RuntimeError(f"Data loading or processing failed for {dataset_id_t}_{y_temp_col}: {str(e)}")

# --------------------------- Model Definitions ---------------------------
# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size, num_conv_layers=3, dropout=0.5, initial_out_channels=32):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        out_channels = initial_out_channels # Configurable initial channels

        # Calculate the maximum possible pooling layers without reducing dimension to 0
        # Each pool halves the size. Need input_size >= 2**num_layers for num_layers pools
        max_valid_layers = int(np.floor(np.log2(input_size))) if input_size > 0 else 0
        if num_conv_layers > max_valid_layers:
             print(f"Warning: Requested {num_conv_layers} conv/pool layers, but input size {input_size} only supports {max_valid_layers}. Reducing layers.")
             num_conv_layers = max(1, max_valid_layers) # Ensure at least 1 layer if possible

        current_size = input_size
        current_out_channels = initial_out_channels

        for i in range(num_conv_layers):
            # Use current_out_channels for this layer's output
            self.conv_layers.append(
                nn.Conv1d(in_channels, current_out_channels, kernel_size=3, padding=1)
            )
            in_channels = current_out_channels # Next layer's input is this layer's output

            # Double channels for the next layer, cap at a reasonable number (e.g., 256 or 512)
            if i < num_conv_layers - 1:
                 current_out_channels = min(current_out_channels * 2, 256) # Cap at 256

            # Simulate pooling to track spatial dimension
            current_size = current_size // 2 # Pool reduces size by half

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size after all conv and pool layers
        spatial_dim_after_pooling = input_size // (2 ** num_conv_layers)
        if spatial_dim_after_pooling == 0:
             # This should be caught by the initial check, but double check
             raise ValueError(f"Calculated spatial dimension after pooling is 0. Input size {input_size}, layers {num_conv_layers}")

        final_out_channels = in_channels # in_channels holds the output channels of the *last* conv layer

        # Adjusted FC layer calculation based on final spatial dimension and channels
        self.fc1 = nn.Linear(final_out_channels * spatial_dim_after_pooling, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1) # Output layer for regression
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, F] (Batch size, Features)
        x = x.unsqueeze(1)  # [B, 1, F] - Add channel dimension for Conv1d

        for i, conv in enumerate(self.conv_layers):
            x = torch.relu(conv(x))
            if x.size(-1) > 1: # Only pool if the dimension is > 1
                 x = self.pool(x)
                 
        # Flatten the output from convolutional layers
        # x shape is now [B, final_out_channels, spatial_dim_after_pooling]
        x = torch.flatten(x, start_dim=1) # [B, final_out_channels * spatial_dim_after_pooling]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) 
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # Output layer (size 1)

        return x

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, num_hidden_layers=2, dropout=0.5):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        in_features = input_size

        # Aim for sizes similar to Script 2's MLP (128, 64) for 2 layers, scale for others.
        layer_sizes = [max(64, input_size // (2**(i+1))) for i in range(num_hidden_layers)]
        if num_hidden_layers >= 1:
             layer_sizes[0] = max(128, layer_sizes[0])

        current_in_features = input_size
        for i in range(num_hidden_layers):
             out_features = layer_sizes[i] 
             out_features = max(1, out_features)
             self.fc_layers.append(nn.Linear(current_in_features, out_features))
             current_in_features = out_features

        self.fc_out = nn.Linear(current_in_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, F]
        for fc in self.fc_layers:
            x = torch.relu(fc(x))
            x = self.dropout(x) # Apply dropout after ReLU
        return self.fc_out(x) # Output layer

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers=2, dropout=0.1, d_model=64, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)

        if d_model % nhead != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        # TransformerEncoder expects input shape (batch_size, seq_len, d_model)
        # input is (batch_size, input_size), after embedding (batch_size, d_model)
        # seq_len = 1 -> (batch_size, 1, d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model) 
        self.dropout_out = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [B, F]
        x = self.embedding(x) # [B, d_model]
        x = x.unsqueeze(1) # [B, 1, d_model]

        x = self.transformer_encoder(x) # Transformer operates on [B, 1, d_model], outputs [B, 1, d_model]

        x = x.squeeze(1) # [B, d_model]

        x = self.norm(x) # [B, d_model]
        x = self.dropout_out(x) #  [B, d_model]

        x = self.fc(x) #  [B, 1]

        return x

# --------------------------- Hyperparameter Distributions ---------------------------
# Includes learning rate, dropout, batch size, and model-specific parameters
models_to_train = {
    'CNN': (CNNModel, {
        'num_conv_layers': optuna.distributions.IntDistribution(2, 4),
        'dropout': optuna.distributions.FloatDistribution(0.1, 0.7),
        'lr': optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        'batch_size': optuna.distributions.IntDistribution(64, 128, log=True),
        'initial_out_channels': optuna.distributions.CategoricalDistribution([16, 32, 64]),
    }),
    'MLP': (MLPModel, {
        'num_hidden_layers': optuna.distributions.IntDistribution(1, 3),
        'dropout': optuna.distributions.FloatDistribution(0.1, 0.7),
        'lr': optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        'batch_size': optuna.distributions.IntDistribution(64, 128, log=True),
    }),
    'Transformer': (TransformerModel, {
        'num_layers': optuna.distributions.IntDistribution(1, 3),
        'dropout': optuna.distributions.FloatDistribution(0.05, 0.3),
        'lr': optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
        'batch_size': optuna.distributions.IntDistribution(64, 128, log=True),
        'd_model': optuna.distributions.CategoricalDistribution([32, 64, 128]),
        'nhead': optuna.distributions.CategoricalDistribution([2, 4, 8]), # Ensure d_model % nhead == 0 during trial selection or validation
    })
}

# --------------------------- Custom PyTorch Regressor (with DataLoader) ---------------------------
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    # Accept lr, num_epochs, batch_size, scaler_y, and model-specific parameters
    def __init__(self, model_class, input_size, lr=1e-3, num_epochs=100, batch_size=64, scaler_y=None, **model_specific_params):
        # Store parameters for get_params
        self.model_class = model_class
        self.input_size = input_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.scaler_y = scaler_y # Store the fitted scaler
        self.model_specific_params = model_specific_params

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.training_time = 0.0
        self.prediction_time = 0.0


    def fit(self, X, y):

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
             raise TypeError("Input X and y must be NumPy arrays.")

        # Convert NumPy arrays to PyTorch tensors and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Create TensorDataset and DataLoader for minibatch training
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model - Pass model_specific_params from __init__
        self.model = self.model_class(
            input_size=self.input_size,
            **self.model_specific_params # Pass other stored parameters to model __init__
        ).to(self.device)

        criterion = nn.MSELoss() # Use MSELoss as y is scaled
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop (with minibatches and exception protection)
        start_time = time.time()
        # print(f"Starting training with {self.num_epochs} epochs, batch size {self.batch_size} on {self.device}") # Optional: verbose during fit
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                # Use squeeze() to match shape [batch_size] vs [batch_size, 1]
                loss = criterion(outputs.squeeze(-1), batch_y)

                # Process Loss exception
                if torch.isnan(loss):
                    print(f"Epoch {epoch}, Batch {num_batches}: Loss is NaN, stopping training early")
                    # Clean up potential CUDA memory
                    del batch_x, batch_y, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Raise an exception to signal failure to OptunaSearchCV
                    raise RuntimeError("NaN loss encountered during training")

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            # if num_batches > 0:
            #     avg_loss = total_loss / num_batches
            #     print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}')


        self.training_time = time.time() - start_time
        # print(f"Training finished in {self.training_time:.2f} seconds.") 

        # Clean up memory after training
        del X_tensor, y_tensor, train_dataset, train_loader 
        torch.cuda.empty_cache() 
        gc.collect() 

        return self

    def predict(self, X):
        # X is expected to be unscaled NumPy array (features)
        start_time = time.time()
        if not isinstance(X, np.ndarray):
             raise TypeError("Input X must be a NumPy array.")

        # Convert test data to tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval() 
        with torch.no_grad(): 
            y_pred_tensor = self.model(X_tensor)
            # Return scaled prediction as NumPy array
            y_pred_scaled = y_pred_tensor.squeeze(-1).cpu().numpy()

        self.prediction_time = time.time() - start_time

        # Clean up memory after prediction
        del X_tensor, y_pred_tensor
        torch.cuda.empty_cache()
        gc.collect()

        # Return scaled predictions. Inverse scaling happens in measure_performance.
        return np.nan_to_num(y_pred_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Scikit-learn compatibility methods ---
    def get_params(self, deep=True):
        # Return parameters that can be set by OptunaSearchCV
        # scaler_y is NOT a parameter to be tuned or set by OptunaSearchCV
        params = {
            'model_class': self.model_class, # Required for cloning by OptunaSearchCV
            'input_size': self.input_size,   # Required for cloning
            'lr': self.lr,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
        }
        # Add model specific parameters
        params.update(self.model_specific_params)
        return params

    def set_params(self, **params):
        # Set parameters provided by OptunaSearchCV
        valid_params = self.get_params(deep=True) # Get all potential params including model_specific

        for param, value in params.items():
            if param in ['model_class', 'input_size', 'scaler_y']:
                 continue
            elif hasattr(self, param):
                # Update core parameters like lr, num_epochs, batch_size
                setattr(self, param, value)
            elif param in self.model_specific_params or param in models_to_train.get(self.model_class.__name__, (None, {}))[1]:
                 # Update parameters intended for the model_specific_params dict
                 self.model_specific_params[param] = value
            # else:
            #     # print(f"Warning: Parameter '{param}' not recognized by PyTorchRegressor.set_params") # Optional warning
        return self

# --------------------------- Performance Evaluation ---------------------------
def measure_performance(model, X_test, y_test_scaled, metrics):
    """
    Calculates model performance metrics.
    X_test is unscaled features, y_test_scaled is scaled target.
    Predictions are inversely scaled before metrics calculation.
    """

    y_test_scaled = np.asarray(y_test_scaled).flatten()

    # model.predict returns scaled predictions
    mem_profile = memory_usage((model.predict, (X_test,)), interval=0.1, timeout=600, retval=True)
    if isinstance(mem_profile, tuple):
        # mem_profile is (list_of_memory_snapshots, return_value_of_predict)
        mem_list, y_pred_scaled = mem_profile
        max_mem = max(mem_list) if mem_list else 0.0
    else:
        # Profiling failed (e.g., timeout), call predict directly
        print("Warning: memory_usage profiling failed or timed out during prediction.")
        y_pred_scaled = model.predict(X_test)
        max_mem = 0.0 

    # --- Inverse standardize predictions and true values ---
    scaler_y = model.scaler_y
    if scaler_y is None:
        print("Error: scaler_y is not available in the model. Cannot inverse standardize.")
        # Fallback: Calculate metrics on scaled data or return NaNs
        # Returning NaNs is safer as the scale is unknown
        metrics.update({
             'Training Time (seconds)': model.training_time,
             'Prediction Time (seconds)': model.prediction_time,
             'Max Memory Usage (MiB)': max_mem,
             'RMSE': np.nan, 'PCC': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC/RMSE': np.nan
        })
        return metrics

    try:
        # Inverse transform expects 2D arrays, then flatten back to 1D
        y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    except Exception as e:
         print(f"Error during inverse transformation: {e}")
         # Return NaNs if inverse transformation fails
         metrics.update({
             'Training Time (seconds)': model.training_time,
             'Prediction Time (seconds)': model.prediction_time,
             'Max Memory Usage (MiB)': max_mem,
             'RMSE': np.nan, 'PCC': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC/RMSE': np.nan
         })
         return metrics


    # Handle extreme cases (e.g., no samples) after inverse transform
    if len(y_test) == 0 or len(y_pred) == 0:
         print("Warning: Test set or prediction results are empty after inverse transform.")
         metrics.update({
             'Training Time (seconds)': model.training_time,
             'Prediction Time (seconds)': model.prediction_time,
             'Max Memory Usage (MiB)': max_mem,
             'RMSE': np.nan, 'PCC': np.nan, 'MAE': np.nan, 'R2': np.nan, 'PCC/RMSE': np.nan
         })
         return metrics
    if len(y_test) != len(y_pred):
         print(f"Warning: Mismatch between y_test ({len(y_test)}) and y_pred ({len(y_pred)}) lengths after inverse transform. Truncating.")
         min_len = min(len(y_test), len(y_pred))
         y_test = y_test[:min_len]
         y_pred = y_pred[:min_len]
    # Calculate metrics on inverse scaled data
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    try:
        pcc = pearsonr(y_test, y_pred)[0]
        pcc = pcc if np.isfinite(pcc) else 0.0 # Handle NaN correlation
    except Exception:
        pcc = 0.0 # Correlation calculation failed

    mae = mean_absolute_error(y_test, y_pred)

    try:
        r2 = r2_score(y_test, y_pred)
        r2 = r2 if np.isfinite(r2) else 0.0 # Handle NaN R2
    except Exception:
        r2 = 0.0 # R2 calculation failed

    # # Handle division by zero for PCC/RMSE
    # pcc_rmse = pcc / rmse if rmse != 0 else (0.0 if pcc == 0 else np.inf * np.sign(pcc))

    metrics.update({
        'Training Time (seconds)': model.training_time,
        'Prediction Time (seconds)': model.prediction_time,
        'Max Memory Usage (MiB)': max_mem,
        'RMSE': rmse,
        'PCC': pcc,
        'MAE': mae,
        'R2': r2,
        # 'PCC/RMSE': pcc_rmse
    })
    return metrics

def pcc_scorer(y_true_scaled, y_pred_scaled):
    """
    Safe PCC scoring function for OptunaSearchCV.
    Operates on the SCALED values provided during cross-validation.
    Handles zero variance cases.
    """
    y_true_scaled = np.asarray(y_true_scaled).flatten()
    y_pred_scaled = np.asarray(y_pred_scaled).flatten()

    if len(y_true_scaled) == 0 or len(y_pred_scaled) == 0:
         return 0.0 # Return 0 score for empty data

    if np.var(y_true_scaled) < 1e-9 or np.var(y_pred_scaled) < 1e-9:
        return 0.0 # Variance too small considered invalid data, correlation is 0

    # Calculate Pearson correlation
    try:
        corr = pearsonr(y_true_scaled, y_pred_scaled)[0]
    except Exception:
        corr = np.nan

    # Returning -1.0 penalizes trials that result in invalid correlations
    return corr if np.isfinite(corr) else -1.0

# --------------------------- Model Saving ---------------------------
def save_model(model, dataset_id, model_name, metrics, save_path):
    """Saves the model state dictionary and performance metrics."""
    save_dir = os.path.join(save_path, dataset_id, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 'model' here is the PyTorchRegressor instance
    if model.model is not None:
        try:
            # Save the actual PyTorch model's state_dict
            torch.save(model.model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
        except Exception as e:
            print(f"Warning: Failed to save PyTorch model state_dict for {model_name}: {e}")
    else:
        print(f"Warning: PyTorch model for {model_name} in {dataset_id} is None, cannot save state_dict.")

    final_params = model.get_params()
    # Filter out non-serializable params like 'model_class' and 'scaler_y' (which is a type/object)
    serializable_params = {k: v for k, v in final_params.items() if k not in ['model_class'] and not isinstance(v, (type, StandardScaler))}

    # Also save the fitted scaler_y object using joblib
    scaler_path = os.path.join(save_dir, "scaler_y.joblib")
    if model.scaler_y is not None:
        try:
            joblib.dump(model.scaler_y, scaler_path)
        except Exception as e:
            print(f"Warning: Failed to save scaler_y for {model_name}: {e}")
    else:
        print(f"Warning: scaler_y is None for {model_name}, cannot save scaler.")


    metadata_path = os.path.join(save_dir, "metadata.joblib")
    try:
        joblib.dump({
            'model_class_name': model.model_class.__name__, # Save class name
            'final_params': serializable_params, # Save parameters
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, metadata_path)
        print(f"Model state_dict, scaler, and metadata saved to: {save_dir}")
    except Exception as e:
         print(f"Warning: Failed to save metadata for {model_name}: {e}")


# --------------------------- Main Process ---------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Define the params
    # set the optunacv search params
    n_splits = 10 # Number of cross-validation folds during hyperparameter optimization for base models
    n_trials = 20 # Number of iterations for hyperparameter optimization
    # save path
    results_base_path = "checkpoints" 
    os.makedirs(results_base_path, exist_ok=True)

    # Iterate through all datasets and target variables
    for dataset_id_t in [k.replace('_pheno', '') for k in dataset_paths if '_pheno' in k]:
        pheno_key = f"{dataset_id_t}_pheno"
        if pheno_key not in pheno_dict:
             print(f"Warning: Pheno key '{pheno_key}' not found in pheno_dict. Skipping dataset '{dataset_id_t}'.")
             continue

        target_cols = pheno_dict.get(pheno_key, []) # Get target columns for this dataset

        for y_temp_col in target_cols:
            task_data = f"{dataset_id_t}_{y_temp_col}"
            print(f"\n===== Processing {task_data} =====")

            # Data loading, splitting, and standardization (with exception handling)
            try:
                # get_data returns: unscaled X_train/X_test, scaled y_train/y_test, and scaler_y
                X_train, X_test, y_train_scaled, y_test_scaled, scaler_y = get_data(dataset_id_t, y_temp_col)
                input_size = X_train.shape[1]
                print(f"Successfully loaded data for {task_data}. X_train shape: {X_train.shape}, y_train_scaled shape: {y_train_scaled.shape}")
            except Exception as e:
                print(f"Data error for {task_data}: {str(e)}")
                continue # Skip this task if data loading/processing fails

            results = [] # To store metrics for different models on this task

            # --------------------------- Hyperparameter Optimization ---------------------------
            # models_to_train includes CNN, MLP, Transformer based on the dictionary
            for model_name, (model_cls, param_dist) in models_to_train.items():
                print(f"\n--- Optimizing {model_name} ---")

                # Define base estimator for OptunaSearchCV
                base_estimator = PyTorchRegressor(
                    model_class=model_cls,
                    input_size=input_size,
                    num_epochs=50, 
                    scaler_y=scaler_y, # Pass the scaler
                    # Pass potential model specific params needed by __init__
                    **{k: (param_dist[k].low if hasattr(param_dist[k], 'low') else (param_dist[k].choices[0] if hasattr(param_dist[k], 'choices') else None)) for k in param_dist if k not in ['lr', 'num_epochs', 'batch_size']}
                )

                try:
                    # Explicitly create study object with a unique name per task+model
                    study_name = f"{task_data.replace('/', '_').replace('.', '_')}_{model_name}_study" # Create a safe study name
                    study = optuna.create_study(direction='maximize', study_name=study_name, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5), load_if_exists=True)

                    print(f"Optuna study '{study_name}' created or loaded.")

                    optuna_search = OptunaSearchCV(
                        estimator=base_estimator,
                        param_distributions=param_dist,
                        cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), # Use 10 folds for CV during tuning
                        n_trials=n_trials, 
                        scoring=make_scorer(pcc_scorer), # Maximize PCC (calculated on scaled data)
                        n_jobs=1,
                        random_state=42,
                        study=study, 
                        verbose=2 
                    )

                    print(f"Starting {model_name} optimization with max {optuna_search.n_trials} trials, using {optuna_search.cv.n_splits}-fold CV.")
                    # OptunaSearchCV fits on the full X_train, y_train_scaled data (NumPy arrays)
                    optuna_search.fit(X_train, y_train_scaled)

                    # Get best parameters found by Optuna
                    best_params = optuna_search.best_params_
                    print(f"Optimization finished for {model_name}. Best params: {best_params}")

                except Exception as e:
                    print(f"Optimization for {model_name} failed with exception: {str(e)}")
                    print(f"Falling back to original parameters for {model_name}: {ORIGINAL_HYPERPARAMS.get(model_name, {})}")
                    best_params = ORIGINAL_HYPERPARAMS.get(model_name, {})
                    if not best_params:
                        print(f"Warning: No original parameters found for {model_name}. Skipping final training.")
                        continue # Skip final training if no params are available

                # --- Final Training with Best (or Original) Parameters ---
                # Use the num_epochs, lr, batch_size from the best params (or original) for the final training
                final_num_epochs = best_params.get('num_epochs', ORIGINAL_HYPERPARAMS.get(model_name, {}).get('num_epochs', 100))
                final_lr = best_params.get('lr', ORIGINAL_HYPERPARAMS.get(model_name, {}).get('lr', 1e-3))
                final_batch_size = best_params.get('batch_size', ORIGINAL_HYPERPARAMS.get(model_name, {}).get('batch_size', 64))

                print(f"\nTraining final {model_name} model for {final_num_epochs} epochs with lr={final_lr}, batch_size={final_batch_size} and other best params...")

                model_specific_final_params = {k: v for k, v in best_params.items() if k not in ['lr', 'num_epochs', 'batch_size']}

                final_model = PyTorchRegressor(
                    model_class=model_cls,
                    input_size=input_size,
                    lr=final_lr,
                    num_epochs=final_num_epochs,
                    batch_size=final_batch_size,
                    scaler_y=scaler_y, 
                    **model_specific_final_params 
                )
                final_model.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
                print(f"Using device: {final_model.device}")

                try:
                    # Train the final model on the entire training set (X_train (unscaled), y_train_scaled)
                    final_model.fit(X_train, y_train_scaled)
                    print(f"Final model training complete for {model_name}.")
                except RuntimeError as e:
                    print(f"Final training failed due to runtime error for {model_name}: {str(e)}")
                    continue # Skip evaluation and saving if training fails

                # --------------------------- Performance Evaluation and Saving ---------------------------
                # Calculate performance metrics on the test set
                # Pass X_test (unscaled) and y_test_scaled. measure_performance handles inverse scaling.
                metrics = {}
                metrics['Model'] = model_name
                metrics['Dataset'] = task_data
                metrics = measure_performance(final_model, X_test, y_test_scaled,metrics)
                metrics['Best Params'] = best_params
                results.append(metrics) # Add to results list
                print(f"Performance metrics for {model_name} on {task_data} (on original scale):")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                         print(f"  {metric}: {value:.4f}")
                    else:
                         print(f"  {metric}: {value}")


                # Save model, scaler, and metadata
                save_model(
                    model=final_model, # Pass the PyTorchRegressor instance
                    dataset_id=task_data,
                    model_name=model_name,
                    metrics=metrics,
                    save_path=results_base_path
                )

                # Clean up memory after processing each model
                del final_model 
                torch.cuda.empty_cache()
                gc.collect()

            # Save performance results for this task (after processing all models for this task)
            if results:
                results_df = pd.DataFrame(results)
                output_path = os.path.join(results_base_path, task_data, "DL_performance.xlsx")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                results_df.to_excel(output_path, index=False)
                print(f"Results for {task_data} saved to: {output_path}")
            else:
                print(f"No valid results generated for {task_data}")

            # Clear memory after each task (including scaler_y)
            del X_train, X_test, y_train_scaled, y_test_scaled, scaler_y
            gc.collect()
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()

    print("\n===== All specified tasks finished =====")
