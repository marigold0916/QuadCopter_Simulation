import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ResidualDataset(Dataset):
    def __init__(self,X,Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ResidualDynamicNet(nn.Module):
    def __init__(self, input_dim = 16, output_dim = 6):
        super(ResidualDynamicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128,128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim)
        ) 
    def forward(self,x):
        return self.net(x)

def main():
    print("Data Load")
    if not (os.path.exists('X_data.npy') and os.path.exists('Y_data.npy')):
        print("가상 데이터를 생성")
        X = np.random.rand(100000,16)
        Y = np.random.rand(100000, 6)
    else:
        X = np.load('X_data.npy')
        Y = np.load('Y_data.npy')
    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Preprocessing & Scaling Data")
    scaler_X      = MinMaxScaler()
    scaler_Y      = MinMaxScaler()
    X_train_scale = scaler_X.fit_transform(X_train)
    Y_train_scale = scaler_Y.fit_transform(Y_train)
    X_val_scale   = scaler_X.transform(X_val)
    Y_val_scale   = scaler_Y.transform(Y_val)
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_Y, 'scaler_Y.pkl')
    print("Saved 'scaler_X.pkl' and 'scaler_Y.pkl'")

    batch_size = 64
    train_data = ResidualDataset(X_train_scale, Y_train_scale)
    val_data   = ResidualDataset(X_val_scale, Y_val_scale)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    model = ResidualDynamicNet(input_dim=16, output_dim=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


    epochs = 50
    train_loss = []
    val_loss   = []

    print("\nStart Train")
    best_val_loss = float('inf')

    for epoch in range(1,epochs+1):
        model.train()
        running_train_loss = 0.0
        for inputs, targets  in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs   = model(inputs)
                loss      = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss    = running_val_loss / len(val_loader.dataset)
        val_loss.append(epoch_val_loss) 
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_rdl_model.pth')
        if epoch % 5 == 0 or epoch ==1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.6f}")
    print("Saved 'best_rdl_model.pth")

    plt.figure(figsize=(8,5))
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Normalized Scale)')
    plt.title('Residual Dynamic Learning Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()


    model.eval()
    all_preds       = []
    all_targets     = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    preds_scaled    = np.vstack(all_preds)
    targets_scaled  = np.vstack(all_targets)
    preds_raw       = scaler_Y.inverse_transform(preds_scaled)
    targets_raw     = scaler_Y.inverse_transform(targets_scaled)
    rmse_per_dim    = np.sqrt(np.mean((preds_raw - targets_raw) ** 2, axis=0))
    mae_per_dim     = np.mean(np.abs(preds_raw - targets_raw), axis=0)
    dim_names       = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    print("\n=== Real Physical Unit Performance Evaluation ===")
    for i in range(min(len(dim_names), preds_scaled.shape[1])):
        print(f"[{dim_names[i]:>5}] RMSE: {rmse_per_dim[i]:.6f} | MAE: {mae_per_dim[i]:.6f}")
    total_rmse = np.sqrt(np.mean((preds_raw - targets_raw)**2))
    print(f"Total Mean RMSE: {total_rmse:.6f}")   


if __name__ == "__main__":
    main()