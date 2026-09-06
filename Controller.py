import numpy as np

class Controller:
    def __init__(self, m=1.5, g=9.81,
        Kp_pos = (2.0, 4.0,4.0), Kd_pos = (2.0,2.0,3.0),
        Kp_att = (6.0, 6.0,3.0), Kd_att = (2.0,2.0,1.0)):
        self.m = m; self.g=g
        self.Kp_pos = np.array(Kp_pos); self.Kd_pos = np.array(Kd_pos)
        self.Kp_att = np.array(Kp_att); self.Kd_att = np.array(Kd_att)

    def compute_control(self, state, target_pos, target_yaw=0.0):
        x, y, z , phi, theta, psi, vx, vy, vz, p, q, r =state
        pos = np.array([x,y,z])
        vel = np.array([vx, vy, vz])
        target_pos=np.array(target_pos)

        acc_des = self.Kp_pos*(target_pos - pos) - self.Kd_pos * vel
        acc_des[2] += self.g

        zb_des = acc_des / np.linalg.norm(acc_des)
        xc_des = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        yb_des = np.cross(zb_des, xc_des)
        yb_des /= np.linalg.norm(yb_des)
        xb_des = np.cross(yb_des, zb_des)
        R_des = np.column_stack([xb_des, yb_des, zb_des])

        theta_des = np.arcsin(-R_des[2,0])
        phi_des = np.arctan2(R_des[2,1], R_des[2,2])
        psi_des = np.arctan2(R_des[1,0], R_des[0,0])

        F_total = self.m * np.linalg.norm(acc_des)
        att_err = np.array([phi_des-phi, theta_des-theta,psi_des-psi])
        rate= np.array([p,q,r])
        tau = self.Kp_att*att_err - self.Kd_att *rate
        return F_total, tau[0], tau[1], tau[2]