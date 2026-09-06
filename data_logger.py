import numpy as np
import matplotlib.pyplot as plt
import os

class DataLogger:
    """
    Visualizer의 update 함수를 감싸서(Wrap)
    프로그램이 종료될 때까지 실시간으로 데이터를 계속 수집하고
    창이 닫히면 자동으로 그래프들을 저장합니다 (통합본 + 데이터별 개별 플롯본).
    """
    def __init__(self, filename="quadcopter_simulation_results.png"):
        self.filename = filename
        # 파일 경로 분석을 통해 개별 저장용 접두어(prefix) 생성
        self.base_dir = os.path.dirname(filename)
        self.base_name = os.path.splitext(os.path.basename(filename))[0]
        
        self.log = {
            'time': [],
            'pos': [],      # [x, y, z]
            'vel': [],      # [vx, vy, vz]
            'acc': [],      # [ax, ay, az]
            'thrusts': [],  # [f1, f2, f3, f4]
            'torques': []   # [tau_x, tau_y, tau_z]
        }
        self.prev_vel = np.zeros(3)

    def attach(self, model, controller, target_pos, dt):
        """
        model.steps와 controller.compute_control에 훅(Hook)을 걸어
        3D 애니메이션 실행 중 데이터가 수집되도록 설정합니다.
        """
        orig_compute = controller.compute_control
        orig_steps = model.steps

        # 매 프레임 제어 입력을 가로채서 기록 준비
        def compute_wrapper(state, target_p, *args, **kwargs):
            F, tx, ty, tz = orig_compute(state, target_p, *args, **kwargs)
            self._temp_torques = [tx, ty, tz]
            self._temp_F = F
            return F, tx, ty, tz

        # 매 프레임 물리 연산을 가로채서 데이터 기록
        def steps_wrapper(thrusts, dt_val, *args, **kwargs):
            current_time = len(self.log['time']) * dt_val
            
            prev_v = model.state[6:9].copy()
            state = orig_steps(thrusts, dt_val, *args, **kwargs)
            
            x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = state
            curr_v = np.array([vx, vy, vz])
            
            # 수치 미분 가속도 (dv/dt)
            acc = (curr_v - prev_v) / dt_val

            # 데이터 기록
            self.log['time'].append(current_time)
            self.log['pos'].append([x, y, z])
            self.log['vel'].append([vx, vy, vz])
            self.log['acc'].append(acc)
            self.log['thrusts'].append(thrusts.copy())
            self.log['torques'].append(getattr(self, '_temp_torques', [0, 0, 0]))

            return state

        controller.compute_control = compute_wrapper
        model.steps = steps_wrapper

        # 현재 열려 있는 matplotlib figure를 찾아서 '창 닫힘' 이벤트 연결
        fig = plt.gcf()
        fig.canvas.mpl_connect('close_event', self.on_close)

    def on_close(self, event):
        """3D 시각화 창이 닫힐 때 수집된 전체 데이터로 다양한 버전의 그래프를 저장합니다."""
        if len(self.log['time']) == 0:
            print("\n[알림] 수집된 데이터가 없습니다.")
            return

        time = np.array(self.log['time'])
        pos = np.array(self.log['pos'])
        vel = np.array(self.log['vel'])
        acc = np.array(self.log['acc'])
        thrusts = np.array(self.log['thrusts'])
        torques = np.array(self.log['torques'])

        # =========================================================================
        # 방식 1: 기존 통합 그래프 저장 (모든 성분이 한 plot에 겹쳐 나오는 방식 유지)
        # =========================================================================
        fig, axs = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
        fig.suptitle('Quadcopter Simulation Log Results (Combined)', fontsize=15, fontweight='bold')

        # 1. 위치 (X, Y, Z)
        axs[0].plot(time, pos[:, 0], 'r-', label='X [m]')
        axs[0].plot(time, pos[:, 1], 'g-', label='Y [m]')
        axs[0].plot(time, pos[:, 2], 'b-', label='Z [m]')
        axs[0].set_ylabel('Position [m]')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        # 2. 속도 (Vx, Vy, Vz)
        axs[1].plot(time, vel[:, 0], 'r--', label='Vx [m/s]')
        axs[1].plot(time, vel[:, 1], 'g--', label='Vy [m/s]')
        axs[1].plot(time, vel[:, 2], 'b--', label='Vz [m/s]')
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        # 3. 가속도 (Ax, Ay, Az)
        axs[2].plot(time, acc[:, 0], 'r:', label='Ax [m/s²]')
        axs[2].plot(time, acc[:, 1], 'g:', label='Ay [m/s²]')
        axs[2].plot(time, acc[:, 2], 'b:', label='Az [m/s²]')
        axs[2].set_ylabel('Acc [m/s²]')
        axs[2].legend(loc='upper right')
        axs[2].grid(True)

        # 4. 각 로터 추력 (Rotor 1~4)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i in range(4):
            axs[3].plot(time, thrusts[:, i], label=f'Rotor {i+1}', color=colors[i])
        axs[3].set_ylabel('Thrust [N]')
        axs[3].legend(loc='upper right')
        axs[3].grid(True)

        # 5. 토크 (Tau X, Y, Z)
        axs[4].plot(time, torques[:, 0], 'c-', label='Tau X (Roll)')
        axs[4].plot(time, torques[:, 1], 'm-', label='Tau Y (Pitch)')
        axs[4].plot(time, torques[:, 2], 'y-', label='Tau Z (Yaw)')
        axs[4].set_ylabel('Torque [N·m]')
        axs[4].set_xlabel('Time [s]')
        axs[4].legend(loc='upper right')
        axs[4].grid(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(self.filename, dpi=300)
        plt.close(fig)
        print(f"\n[알림] 기존 통합 그래프가 '{self.filename}'로 저장되었습니다.")

        # =========================================================================
        # 방식 2: 각 성분별로 독립적인 Plot을 만들어서 개별 이미지로 저장하는 기능 추가
        # =========================================================================
        
        # Helper 함수: 3개의 성분(X, Y, Z 등)을 각각의 플롯으로 쪼개서 하나의 이미지로 저장
        def save_separated_3axes(data, labels, colors, title, sub_filename, y_label):
            fig_sep, axs_sep = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig_sep.suptitle(title, fontsize=14, fontweight='bold')
            for idx in range(3):
                axs_sep[idx].plot(time, data[:, idx], color=colors[idx], label=labels[idx])
                axs_sep[idx].set_ylabel(y_label)
                axs_sep[idx].legend(loc='upper right')
                axs_sep[idx].grid(True)
            axs_sep[2].set_xlabel('Time [s]')
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            full_path = os.path.join(self.base_dir, f"{self.base_name}_{sub_filename}.png")
            plt.savefig(full_path, dpi=300)
            plt.close(fig_sep)
            print(f"[알림] 분리된 개별 그래프가 '{full_path}'로 저장되었습니다.")

        # 2-1. 위치 개별 분리 (X / Y / Z 따로)
        save_separated_3axes(pos, ['X [m]', 'Y [m]', 'Z [m]'], ['r', 'g', 'b'], 
                             'Quadcopter Position Results', 'pos', 'Position [m]')

        # 2-2. 속도 개별 분리 (Vx / Vy / Vz 따로)
        save_separated_3axes(vel, ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]'], ['r', 'g', 'b'], 
                             'Quadcopter Velocity Results', 'vel', 'Velocity [m/s]')

        # 2-3. 가속도 개별 분리 (Ax / Ay / Az 따로)
        save_separated_3axes(acc, ['Ax [m/s²]', 'Ay [m/s²]', 'Az [m/s²]'], ['r', 'g', 'b'], 
                             'Quadcopter Acceleration Results', 'acc', 'Acc [m/s²]')

        # 2-4. 토크 개별 분리 (Tau X / Tau Y / Tau Z 따로)
        save_separated_3axes(torques, ['Tau X (Roll)', 'Tau Y (Pitch)', 'Tau Z (Yaw)'], ['c', 'm', 'y'], 
                             'Quadcopter Control Torques Results', 'torques', 'Torque [N·m]')

        # 2-5. 각 로터 추력 개별 분리 (Rotor 1 / 2 / 3 / 4 따로, 4개 플롯 생성)
        fig_thrust, axs_thrust = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig_thrust.suptitle('Quadcopter Rotor Thrusts Results', fontsize=14, fontweight='bold')
        for i in range(4):
            axs_thrust[i].plot(time, thrusts[:, i], color=colors[i], label=f'Rotor {i+1}')
            axs_thrust[i].set_ylabel('Thrust [N]')
            axs_thrust[i].legend(loc='upper right')
            axs_thrust[i].grid(True)
        axs_thrust[3].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        thrust_path = os.path.join(self.base_dir, f"{self.base_name}_thrusts.png")
        plt.savefig(thrust_path, dpi=300)
        plt.close(fig_thrust)
        print(f"[알림] 분리된 개별 그래프가 '{thrust_path}'로 저장되었습니다.\n")


# 외부 호환성을 위한 편의 함수
def start_logging(model, controller, target_pos, dt):
    logger = DataLogger()
    logger.attach(model, controller, target_pos, dt)
    return logger