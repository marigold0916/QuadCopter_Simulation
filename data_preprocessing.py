import os 
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def main():
    print("DATA LOADED")
    if not(os.path.exists('X_data.npy') and os.path.exists("Y_data.npy")):
        print("NO DATA")
        exit()
    X = np.load('X_data.npy')
    Y = np.load('Y_data.npy')
    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("PreProcessing and Scaling Data")
    scaler_X      = MinMaxScaler()
    scaler_Y      = MinMaxScaler()
    X_train_scale = scaler_X.fit_transform(X_train)
    Y_train_scale = scaler_Y.fit_transform(Y_train)
    X_val_scale   = scaler_X.transform(X_val)
    Y_val_scale   = scaler_Y.transform(Y_val) 
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_Y, 'scaler_Y.pkl')
    print("Saved 'scaler_X.pkl' and 'scaler_Y.pkl")
    np.save('X_train_scale.npy', X_train_scale)
    np.save('Y_train_scale.npy', Y_train_scale)
    np.save('X_val_scale.npy', X_val_scale)
    np.save('Y_val_scale.npy', Y_val_scale)

    print("Saved scaled train/val splits (X_train_scale.npy, X_val_scale.npy, " "Y_train_scale.npy, Y_val_scale.npy)")
if __name__ =="__main__":
    main()
    