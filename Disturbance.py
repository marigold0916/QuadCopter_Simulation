import numpy as np
class WindGust:
    #Dryden 돌풍 모델
    # steady_wind: 정상풍(평균 풍속) 벡터 [Wx, Wy, Wz] (world frame, m/s)
    # sigma: 각 축 난류 강도 (m/s)
    # tau: 각 축 시상수 (s) — 클수록 저주파(느리게 변하는) 돌풍
    def __init__(self, steady_wind=(0.0,0.0,0.0), sigma = (1.0,1.0,0.5), tau = (2.0, 2.0, 1.0), seed=None):
        self.steady = np.array(steady_wind, dtype=float)    
        self.sigma = np.array(sigma, dtype=float)#각 방향의 바람세기
        self.tau = np.array(tau, dtype=float)#바람이 변화하는 속도를 결정하는 시상수
        self.gust = np.zeros(3)#바람의 속도
        self.rng = np.random.default_rng(seed)

    def step(self,dt):
        xi = self.rng.standard_normal(3)#랜덤 노이즈
        self.gust += dt * (-self.gust / self.tau) + self.sigma * np.sqrt(2.0*dt/self.tau) * xi
        return self.steady + self.gust#평균풍속+돌풍


class AeroDrag:
    def __init__(self, k_lin=(0.5, 0.5, 0.6), k_rot=(0.2, 0.2, 0.3), k_wind_torque=0.1):
        self.k_lin = np.array(k_lin, dtype=float)#선속도에 대한 공기저항
        self.k_rot = np.array(k_rot, dtype= float)#각속도에 대한 공기저항
        self.k_wind_torque = k_wind_torque

    def force(self, v_rel):
        return -self.k_lin * np.abs(v_rel) * v_rel#이동방향의 반대로 항력 작용

    def torque(self, rate,v_rel = None):
        tau = -self.k_rot * rate
        if v_rel is not None:
            tau += np.cross([0,0,self.k_wind_torque],v_rel)
        return tau

def compute_disturbance(state, wind_model, drag_model, dt):
    #state: QuadCopter.state (12,) [x,y,z,phi,theta,psi,vx,vy,vz,p,q,r]
    #return: (F_dist_world[3], tau_dist_body[3])
    vel = state[6:9]
    rate = state[9:12]

    wind = wind_model.step(dt)
    v_rel= vel - wind
    F_dist = drag_model.force(v_rel)
    tau_dist = drag_model.torque(rate, v_rel)

    return F_dist, tau_dist#World Frame 기준 외란힘과 Body Frame 기준 외란토크