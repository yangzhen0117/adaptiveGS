import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from scipy.stats import pearsonr
import time
import psutil
from memory_profiler import memory_usage

# Define the dataset class
class GenoPhenoDataset(Dataset):
    def __init__(self, geno_data, pheno_data):
        self.geno_data = geno_data.values  # Convert to NumPy Array
        self.pheno_data = pheno_data.values  # Convert to NumPy Array

    def __len__(self):
        return len(self.geno_data)

    def __getitem__(self, idx):
        # Getting data directly from a NumPy array using an integer index
        geno = torch.tensor(self.geno_data[idx], dtype=torch.float32)
        pheno = torch.tensor(self.pheno_data[idx], dtype=torch.float32)
        return geno, pheno

#read data
snp_df = pd.read_excel('gy_geno.xlsx')
genotype_data = snp_df.iloc[:,1:]
pheno_df = pd.read_excel('gy_pheno.xlsx')
phenotype_data =pheno_df.iloc[:,1]

# Data standardization
genotype_data = (genotype_data - np.mean(genotype_data, axis=0)) / np.std(genotype_data, axis=0)
phenotype_data = (phenotype_data - np.mean(phenotype_data)) / np.std(phenotype_data)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(genotype_data, phenotype_data, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Creating a Data Loader
train_dataset = GenoPhenoDataset(X_train, y_train)
test_dataset = GenoPhenoDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Defining CNN Models
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (X_train.shape[1] // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Defining the MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Define the improved Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()

        # The linear layer maps the input per genotype data to higher dimensions
        self.embedding = nn.Linear(input_size, d_model)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected layer for predicting output
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Data Shape Adjustment： (batch_size, input_size) -> (batch_size, seq_len, input_size)
        # Indicates that each input contains only one genotype information
        x = x.unsqueeze(1)  # Adding a dimension(batch_size, 1, input_size)
        # Embedding through linear layers
        x = self.embedding(x)
        x = self.transformer_encoder(x)

        # Take the output of the last moment as a representation of the whole sequence
        x = x.squeeze(1)

        # LayerNorm and Dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # The final prediction is obtained through the fully connected layer
        x = self.fc(x)
        return x

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Memory usage in MiB

# Functions for training and evaluating models
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=5):
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for geno, pheno in train_loader:
            optimizer.zero_grad()
            outputs = model(geno)
            loss = criterion(outputs.squeeze(), pheno)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    execution_time = time.time() - start_time
    max_memory_usage = get_memory_usage()

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for geno, pheno in test_loader:
            outputs = model(geno)
            y_pred.extend(outputs.squeeze().tolist())
            y_true.extend(pheno.tolist())

    pcc = pearsonr(y_true, y_pred)[0]
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc_rmse = pcc / rmse if rmse != 0 else float('inf')

    metrics = {
        'Execution Time (seconds)': execution_time,
        'Max Memory Usage (MiB)': max_memory_usage,
        'RMSE': rmse,
        'PCC': pcc,
        'PCC/RMSE': pcc_rmse,
        'MAE': mae,
        'R2': r2
    }
    return metrics


# Training and Evaluating CNN Models
cnn_model = CNNModel()
cnn_criterion = nn.MSELoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_metrics = train_and_evaluate(cnn_model, train_loader, test_loader, cnn_criterion, cnn_optimizer)

# Training and Evaluating MLP Models
input_size = X_train.shape[1]
mlp_model = MLPModel(input_size)
mlp_criterion = nn.MSELoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
mlp_metrics = train_and_evaluate(mlp_model, train_loader, test_loader, mlp_criterion, mlp_optimizer)

# Training and Evaluating Transformer Models
d_model = 64
nhead = 4
num_layers = 2
transformer_model = TransformerModel(input_size, d_model, nhead, num_layers)
transformer_criterion = nn.MSELoss()
transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
transformer_metrics = train_and_evaluate(transformer_model, train_loader, test_loader, transformer_criterion, transformer_optimizer)

# Save results to Excel
metrics = {
    "Model": ["CNN", "MLP", "Transformer"],
    "Execution Time (seconds)": [cnn_metrics["Execution Time (seconds)"], mlp_metrics["Execution Time (seconds)"], transformer_metrics["Execution Time (seconds)"]],
    "Max Memory Usage (MiB)": [cnn_metrics["Max Memory Usage (MiB)"], mlp_metrics["Max Memory Usage (MiB)"], transformer_metrics["Max Memory Usage (MiB)"]],
    "RMSE": [cnn_metrics["RMSE"], mlp_metrics["RMSE"], transformer_metrics["RMSE"]],
    "PCC": [cnn_metrics["PCC"], mlp_metrics["PCC"], transformer_metrics["PCC"]],
    "PCC/RMSE": [cnn_metrics["PCC/RMSE"], mlp_metrics["PCC/RMSE"], transformer_metrics["PCC/RMSE"]],
    "MAE": [cnn_metrics["MAE"], mlp_metrics["MAE"], transformer_metrics["MAE"]],
    "R2": [cnn_metrics["R2"], mlp_metrics["R2"], transformer_metrics["R2"]]
}

df = pd.DataFrame(metrics)
excel_file = "trmodel_performance_metrics.xlsx"
df.to_excel(excel_file, index=False)
print(f"Performance metrics have been saved to {excel_file}")