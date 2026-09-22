import numpy as np
from Residual_pred import ResidualPredictor

class Controller:
    def __init__(self, m=1.5, g=9.81,
        Kp_pos = (0.5, 1.4,1.5), Ki_pos = (0.05,1.0,1.0), Kd_pos = (0.5,1.5,1.75),
        Kp_att = (1.5, 1.5,0.75), Ki_att = (1.0,1.0,1.0), Kd_att = (0.25,0.25,0.25),
        Ixx = 0.02, Iyy = 0.02, Izz = 0.04,
        residual_predictor: ResidualPredictor = None):

        self.m = m; self.g=g
        self.Kp_pos = np.array(Kp_pos); self.Ki_pos = np.array(Ki_pos); self.Kd_pos = np.array(Kd_pos)
        self.Kp_att = np.array(Kp_att); self.Ki_att = np.array(Ki_att); self.Kd_att = np.array(Kd_att)
        self.I = np.array([Ixx, Iyy, Izz])

        self.pos_err_int = np.zeros(3)
        self.att_err_int = np.zeros(3)

        self.residual_predictor = residual_predictor
        self.prev_thrusts = np.zeros(4)#인과성: 이번 스텝 thrust는 계산 전이라 직전 스텝 값으로 근사

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
        tau_max = 2.0
        tau = np.clip(tau, -tau_max, tau_max)
        return tau

    def compute_kinetic_mapping(self, acc_des, target_yaw):
        max_tilt = self.g * np.tan(np.deg2rad(60))
        acc_des[0:2] = np.clip(acc_des[0:2], -max_tilt, max_tilt)
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
        F_max   = 48.0
        F_total = np.clip(F_total, 0.0, F_max)
        return F_total, target_att



    def compute_control(self, state, target_pos, target_yaw = 0.0, dt=0.01):
        pos = np.array(state[0:3]); att = np.array(state[3:6])
        vel = np.array(state[6:9]); rate = np.array(state[9:12])
        target_pos = np.array(target_pos)

        acc_des = self.compute_position_control(pos, vel, target_pos, dt)

        residual = None
        if self.residual_predictor is not None:
            residual = self.residual_predictor.predict(np.array(state), self.prev_thrusts)
            acc_des = acc_des - residual[0:3]#예측 외란가속도 상쇄(feedforward)

        F_total, target_att = self.compute_kinetic_mapping(acc_des, target_yaw)

        tau = self.compute_attitude_control(att, rate, target_att, dt)

        if residual is not None:
            tau_res = self.I * residual[3:6]#alpha_res -> tau_res (p_dot=(...+tau_x)/Ixx 역변환)
            tau = tau - tau_res
            tau_max = 2.0
            tau = np.clip(tau, -tau_max, tau_max)

        return F_total, tau[0], tau[1], tau[2]

    def update_applied_thrusts(self, thrusts):
        #QuadCopter.mix()로 실제 인가한 thrust를 다음 스텝 residual 예측 입력으로 저장
        self.prev_thrusts = np.array(thrusts, dtype=float)