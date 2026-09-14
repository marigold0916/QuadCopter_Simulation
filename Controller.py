import numpy as np

class Controller:
    def __init__(self, m=1.5, g=9.81,
        Kp_pos = (0.15, 1.4,1.5), Ki_pos = (0,0,0), Kd_pos = (0.5,1.5,1.75),
        Kp_att = (1.5, 1.5,0.75), Ki_att = (0,0,0), Kd_att = (0.25,0.25,0.25)):

        self.m = m; self.g=g
        self.Kp_pos = np.array(Kp_pos); self.Ki_pos = np.array(Ki_pos); self.Kd_pos = np.array(Kd_pos)
        self.Kp_att = np.array(Kp_att); self.Ki_att = np.array(Ki_att); self.Kd_att = np.array(Kd_att)

        self.pos_err_int = np.zeros(3)
        self.att_err_int = np.zeros(3)

    def compute_position_control(self,pos,vel,target_pos,dt):
        pos_err = target_pos - pos

        self.pos_err_int += pos_err * dt

        self.pos_err_int = np.clip(self.pos_err_int, -2.0, 2.0)

        acc_des = (self.Kp_pos * pos_err) + (self.Ki_pos * self.pos_err_int) - (self.Kd_pos * vel)
        return acc_des
    
    def compute_attitude_control(self,att, rate, target_att,dt):
        att_err = target_att - att

        self.att_err_int += att_err *dt#오차 적분 누적

        self.att_err_int = np.clip(self.att_err_int, -1.0, 1.0)#적분항 제한

        tau = (self.Kp_att * att_err) + (self.Ki_att * self.att_err_int) - (self.Kd_att * rate)
        return tau

    def compute_kinetic_mapping(self, acc_des, target_yaw):
        acc_des_g = acc_des.copy()
        acc_des_g[2] += self.g
        acc_des_g[2] = max(acc_des_g[2], 0.1)

        zb_des = acc_des_g / np.linalg.norm(acc_des_g)
        xc_des = np.array([np.cos(target_yaw), np.sin(target_yaw) , 0.0])
        yb_des = np.cross(zb_des, xc_des)
        yb_des /= np.linalg.norm(yb_des)
        xb_des =np.cross(yb_des, zb_des)
        R_des = np.column_stack([xb_des, yb_des, zb_des])

        theta_des = np.arcsin(-R_des[2, 0])
        phi_des = np.arctan2(R_des[2,1], R_des[2,2])
        psi_des = np.arctan2(R_des[1,0], R_des[0,0])
        target_att = np.array([phi_des, theta_des, psi_des])

        F_total = self.m * np.dot(acc_des_g, zb_des)
        return F_total, target_att



    def compute_control(self, state, target_pos, target_yaw = 0.0, dt=0.01):
        pos = np.array(state[0:3]); att = np.array(state[3:6])
        vel = np.array(state[6:9]); rate = np.array(state[9:12])
        target_pos = np.array(target_pos)

        acc_des = self.compute_position_control(pos, vel, target_pos, dt)

        F_total, target_att = self.compute_kinetic_mapping(acc_des, target_yaw)

        tau = self.compute_attitude_control(att, rate, target_att, dt)

        return F_total, tau[0], tau[1], tau[2]