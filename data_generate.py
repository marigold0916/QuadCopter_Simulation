import numpy as np
from Controller import Controller
from QuadCopter import QuadCopter
from Disturbance import WindGust, AeroDrag, compute_disturbance

def generate_dataset(num_samples=100000, dt=0.005):
    model      = QuadCopter()
    controller = Controller()
    wind_model = WindGust(steady_wind=(2.0, 1.0, 0.0), sigma=(0.8, 0.8, 0.3))
    drag_model = AeroDrag()
    
    X_data = []
    Y_data = []
    
    model.reset()
    target_pos = np.random.uniform([-3.0, -3.0, 0.0], [3.0, 3.0, 10.0])
    
    # ==================== 1. 데이터 수집 루프 ====================
    for step in range(num_samples):
        if step % 1000 == 0:
            target_pos = np.random.uniform([-3.0, -3.0, 0.0], [3.0, 3.0, 10.0])
            
        current_state  = model.state.copy()
        F, tx, ty, tz  = controller.compute_control(current_state, target_pos, dt=dt)
        thrusts        = model.mix(F, tx, ty, tz)
        X_samples      = np.concatenate([current_state, thrusts])
        
        # Nominal 물리 계산
        F_total         = np.sum(thrusts)
        tau_x_nom       = np.sum(model.ys * thrusts)
        tau_y_nom       = np.sum(-model.xs * thrusts)  
        tau_z_nom       = model.c_yaw * np.sum(model.spin_dirs * thrusts)
        
        phi, theta, psi = current_state[3:6]
        p, q, r         = current_state[9:12]
        R               = model.rotation_matrix(phi, theta, psi)
        acc_nom         = (R @ np.array([0, 0, F_total])) / model.m - np.array([0, 0, model.g])
        
        Ixx, Iyy, Izz   = model.I
        p_dot_nom       = ((Iyy - Izz) * q * r + tau_x_nom) / Ixx  # Fix: (Iyy - Izz)
        q_dot_nom       = ((Izz - Ixx) * p * r + tau_y_nom) / Iyy
        r_dot_nom       = ((Ixx - Iyy) * p * q + tau_z_nom) / Izz
        alpha_nom       = np.array([p_dot_nom, q_dot_nom, r_dot_nom])
        
        # 외란 및 Actual 물리 계산
        F_dist, tau_dist = compute_disturbance(current_state, wind_model, drag_model, dt)
        acc_act          = acc_nom + (F_dist / model.m)
        alpha_act        = alpha_nom + np.array([tau_dist[0]/Ixx, tau_dist[1]/Iyy, tau_dist[2]/Izz])
        
        Y_sample         = np.concatenate([acc_act - acc_nom, alpha_act - alpha_nom])

        model.steps(thrusts, dt, ext_force=F_dist, ext_torque=tau_dist)
        
        X_data.append(X_samples)
        Y_data.append(Y_sample)

    # ==================== 2. 루프 종료 후 저장 (for문 밖) ====================
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    np.save('X_data.npy', X_data)
    np.save('Y_data.npy', Y_data)
    
    print(f"수집 완료! X shape: {X_data.shape}, Y shape: {Y_data.shape}")
    print("X_data.npy 및 Y_data.npy 저장 완료.")

if __name__ == "__main__":
    generate_dataset(num_samples=100000, dt=0.005)