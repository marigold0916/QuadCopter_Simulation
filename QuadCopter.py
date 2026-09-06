import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_logger import start_logging
from Controller import Controller

class QuadCopter:
    def __init__(self, m=1.5, arm_length=0.5, Ixx=0.02, Iyy=0.02, Izz=0.04, g=9.81,
                 c_yaw=0.02, spin_dirs=(1,-1,1,-1)):
        self.m = m
        self.L = arm_length
        self.I = np.array([Ixx, Iyy, Izz]) #관성 모멘트 
        self.g = g
        self.c_yaw = c_yaw
        self.spin_dirs = np.array(spin_dirs, dtype=float)
        self.arm_angles = np.deg2rad([45, 135, 225, 315])#로터 위치
        self.xs = self.L * np.cos(self.arm_angles)
        self.ys = self.L * np.sin(self.arm_angles)
        A = np.vstack([np.ones(4), self.ys, -self.xs, self.c_yaw * self.spin_dirs])
        self.A = A
        self.A_inv = np.linalg.inv(A)#혼합 행렬,로터4개의 추력을 x,y,z의 토크로 변경
        self.state = np.zeros(12)#상태 변수 12개(위치, 오일러 각도: 롤/피치/요, 선속도, 각속도)

    def reset(self, initial_state=None):
        self.state = np.zeros(12) if initial_state is None else np.array(initial_state, dtype=float)

    @staticmethod
    def rotation_matrix(phi, theta, psi):#회전 행렬 및 좌표 변환
        #Z-Y-X 오일러 회전 행렬: 기체 좌표계(Body frame)의 벡터를 월드 좌표계(World frame)로 변환
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi),  np.cos(psi), 0],
                        [0, 0, 1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        return Rz @ Ry @ Rx

    @staticmethod
    def transform(points, R, t):
        #점들의 집합에 회전($R$)과 평행이동($t$)을 적용하여 3D 공간상의 위치를 계산
        return (R @ points.T).T + t

    def mix(self, F_total, tau_x, tau_y, tau_z):#제어 명령 변환
        #각 로터가 내야 하는 개별 추력 구함
        cmd = self.A_inv @ np.array([F_total, tau_x, tau_y, tau_z])
        return np.maximum(cmd, 0.0)

    def steps(self, thrusts, dt): # dt 동안 드론의 상태 변화를 계산합니다.
        #로터의 추력으로부터 전체 추력과 3축 토크를 계산
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = self.state
        F_total = np.sum(thrusts)
        tau_x = np.sum(self.ys * thrusts)
        tau_y = np.sum(-self.xs * thrusts)
        tau_z = self.c_yaw * np.sum(self.spin_dirs * thrusts)

        R = self.rotation_matrix(phi, theta, psi)
        acc_world = (R @ np.array([0, 0, F_total])) / self.m - np.array([0, 0, self.g])#선가속도
        #(오일러-뉴턴 방정식)
        ax, ay, az = acc_world

        Ixx, Iyy, Izz = self.I
        p_dot = ((Iyy - Izz) * q * r + tau_x) / Ixx#각가속도 (오일러 회전 방정식)
        q_dot = ((Izz - Ixx) * p * r + tau_y) / Iyy
        r_dot = ((Ixx - Iyy) * p * q + tau_z) / Izz

        phi_dot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        theta_dot = np.cos(phi) * q - np.sin(phi) * r#오일러 각 변화율
        psi_dot = (np.sin(phi) / np.cos(theta)) * q + (np.cos(phi) / np.cos(theta)) * r
#수치 적분 (Euler Integration)
        vx += ax*dt; vy += ay*dt; vz += az*dt
        x += vx*dt; y += vy*dt; z += vz*dt
        p += p_dot*dt; q += q_dot*dt; r += r_dot*dt
        phi += phi_dot*dt; theta += theta_dot*dt; psi += psi_dot*dt

        self.state = np.array([x, y, z, phi, theta, psi, vx, vy, vz, p, q, r])
        return self.state


class Visualizer:
    def __init__(self, arm_length=0.5, r_rotor=0.15):
        self.L = arm_length
        self.r_rotor = r_rotor
        arm_angles = np.deg2rad([45, 135, 225, 315])
        self.arm_tips_body = np.array(
            [[self.L*np.cos(a), self.L*np.sin(a), 0.0] for a in arm_angles]
        )
        n_circle = 20
        theta_c = np.linspace(0, 2*np.pi, n_circle)
        self.rotor_local = np.stack(
            [self.r_rotor*np.cos(theta_c), self.r_rotor*np.sin(theta_c), np.zeros(n_circle)],
            axis=1,
        )

    def live(self, model, controller, target_pos, dt):
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')#1행 1열의 1번째 서브플롯 영역을 만들고 3d좌표계사용
        ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_zlim(0,12)
        ax.set_xlabel('X[m]'); ax.set_ylabel('Y[m]'); ax.set_zlabel('Z[m]')
        ax.set_title('Quadcopter Simulation')

        trail_line = ax.plot([],[],[], "b--", linewidth=1, label="Trajectory")[0]
        #데이터 변경을 위해 리스트의 첫번째 값 꺼내기 
        arm_line = [ax.plot([],[],[], "k--", linewidth=2)[0] for _ in range(4)]
        #드론의 중심에서 4개 로터로 뻗어나가는 암
        rotor_line = [ax.plot([],[],[], 'r-', linewidth=1.5)[0] for _ in range(4)]
        #각 로터 위치에 그려질 빨간색 실선
        center_points, = ax.plot([],[],[], 'ko', markersize=4)
        #길이가 1인 리스트의 원소를 바로 변수에 할당
        ax.legend(loc="upper left")

        trail_x, trail_y, trail_z = [], [], []
        #프레임이 진행됨에 따라 드론이 지나온 X, Y, Z 좌표를 누적하여 저장할 빈 리스트들

        def update(_):
            F, tx, ty, tz = controller.compute_control(model.state, target_pos)
            #현재 기체 상태와 목표 위치로 부터 필요한 총 추력과 3축 토크 계산
            thrusts = model.mix(F, tx, ty, tz)
            #필요한 힘/토크를 4개 개별 로터의 추력으로 분배
            state = model.steps(thrusts, dt)
            # 물리 뉴턴-오일러 방정식을 dt 동안 적분하여 12개 상태량을 update
            x, y, z, phi, theta, psi = state[0:6]
            #12개 상태량 중 앞의 6개 요소인 위치 (x, y, z) 및 오일러 각도 (phi, theta, psi)만 추출
            t = np.array([x, y, z])
            R = model.rotation_matrix(phi, theta, psi)
            #평행이동 벡터 t와 Z-Y-X 오일러 회전 행렬 R을 만듬

            trail_x.append(x); trail_y.append(y); trail_z.append(z)
            trail_line.set_data(trail_x, trail_y)
            #2D 좌표 $(X, Y)$는 set_data()로 설정
            trail_line.set_3d_properties(trail_z)
            #$Z$ 좌표는 별도로 set_3d_properties() 메서드를 호출

            center_points.set_data([x], [y])
            center_points.set_3d_properties([z])
            #현재 드론 중심 단 한 점의 위치만 표시하므로 리스트 형태 [x], [y], [z]로 넣기

            tips_world = model.transform(self.arm_tips_body, R, t)
            #드론 중심이 [0,0,0]에 있고 기울어지지 않은 상태일 때 4개 암 끝점의 상대 좌표
            #기체가 회전(R)하고 이동(t)했을 때 4개 암 끝점의 절대 월드 좌표
            for i in range(4):
                #중심 위치 (x, y, z)부터 i번째 암 끝점 위치(tips_world[i])까지 연결하는 선분을 생성
                arm_line[i].set_data([x, tips_world[i,0]], [y, tips_world[i,1]])
                arm_line[i].set_3d_properties([z, tips_world[i,2]])

            for i in range(4):
                disk_local = self.rotor_local + self.arm_tips_body[i]
                #i번째 암 끝점(self.arm_tips_body[i])으로 원형 프로펠러 지오메트리를 이동
                disk_world = model.transform(disk_local, R, t)
                #기체의 전체 회전(R)과 현재 위치(t)를 적용하여 로터 원의 20개 점을 월드 좌표계로 강체 변환
                rotor_line[i].set_data(disk_world[:,0], disk_world[:,1])
                #20개 점의 X, Y, Z 좌표 슬라이싱 배열을 전달하여 3D 원을 완성
                rotor_line[i].set_3d_properties(disk_world[:,2])

            return [trail_line, center_points] + arm_line + rotor_line

        anim = FuncAnimation(fig, update, frames=200, interval=dt*1000, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    print("Starting Quadcopter Simulation...")
    arm_length = 0.5
    r_rotor = 0.15
    m = 1.5
    Ixx, Iyy, Izz = 0.02, 0.02, 0.04
    g = 9.81
    c_yaw = 0.02
    dt = 0.02
    target_pos = [2.7, -1.3, 11.1]

    model = QuadCopter(m=m, arm_length=arm_length, Ixx=Ixx, Iyy=Iyy, Izz=Izz, g=g, c_yaw=c_yaw)
    controller = Controller(m=m, g=g)
    visualizer = Visualizer(arm_length=arm_length, r_rotor=r_rotor)
    visualizer.live(model, controller, target_pos, dt)

    start_logging(model, controller, target_pos, dt)
    visualizer.live(model, controller, target_pos, dt)