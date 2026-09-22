import numpy as np
import torch
import joblib
from model import ResidualDynamicNet

class ResidualPredictor:
    def __init__(self, model_path = 'best_rdl_model.pth', scaler_x_path = 'scaler_X.pkl', 
                 scaler_y_path = 'scaler_Y.pkl', input_dim = 16, output_dim = 6, device='cpu'):
        self.device = torch.device(device)
        self.model  = ResidualDynamicNet(input_dim=input_dim, output_dim=output_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.scale_X = joblib.load(scaler_x_path)
        self.scale_Y = joblib.load(scaler_y_path)

    @torch.no_grad()
    def predict(self, state, thrusts):
        x_raw    = np.concatenate([state, thrusts]).astype(np.float32).reshape(1,-1)
        x_scaled = self.scale_X.transform(x_raw)
        x_t      = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        y_scaled = self.model(x_t).cpu().numpy()
        residual_raw = self.scale_Y.inverse_transform(y_scaled).reshape(-1)
        return residual_raw