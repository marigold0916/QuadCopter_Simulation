# Quadrotor Simulation & Residual Dynamics Learning

Python으로 구현한 쿼드콥터 시뮬레이션 및 제어 프로젝트입니다.

6-DOF 쿼드콥터 동역학 모델과 PID 제어기를 구현한 뒤, 외란 모델링과 폐루프 안정성 분석을 수행하고, **Residual Dynamics Learning**을 이용한 데이터 기반 외란 보상을 실험합니다.

```text
Quadrotor Dynamics
        ↓
PID Control
        ↓
Disturbance Modeling
        ↓
Stability Analysis
        ↓
Residual Dynamics Learning
        ↓
Data-driven Compensation
```

> **Current Status:** The residual dynamics model has been integrated into the controller, but the overall control performance is currently under investigation.

---

## 1. Overview

이 프로젝트의 목적은 물리 기반 모델과 데이터 기반 모델을 결합하여 외란이 존재하는 동적 시스템을 분석하고 제어하는 과정을 구현하는 것입니다.

프로젝트는 다음 단계로 진행되었습니다.

1. 쿼드콥터 6-DOF 동역학 모델 구현
2. PID 기반 위치 및 자세 제어 구현
3. Wind Gust 및 Aerodynamic Drag 모델링
4. 제어기의 선형 안정성 및 pole 분석
5. Residual Dynamics dataset 생성
6. Neural Network 기반 residual dynamics 학습
7. 학습된 모델을 이용한 disturbance compensation

---

## 2. Quadrotor Dynamics & PID Control

쿼드콥터의 상태는 다음과 같이 정의합니다.

$$
x =
[x,y,z,\phi,\theta,\psi,v_x,v_y,v_z,p,q,r]^T
$$

* Position: \(x,y,z\)
* Attitude: \(\phi,\theta,\psi\)
* Linear velocity: \(v_x,v_y,v_z\)
* Angular velocity: \(p,q,r\)

Z-Y-X Euler rotation을 사용하여 body frame과 world frame 사이의 변환을 수행합니다.

각 rotor의 thrust를 전체 thrust와 body torque로 변환하기 위해 rotor mixing을 사용합니다.

```text
Rotor Thrusts
     ↓
Rotor Mixer
     ↓
Total Thrust + Body Torques
     ↓
Quadrotor Dynamics
     ↓
State Update
```

### PID Controller

위치 제어기는 position error를 이용하여 desired acceleration을 계산합니다.

$$
a_{cmd}
=
K_p e
+
K_i\int e\,dt
+
K_d\dot e
$$

이후 desired acceleration으로부터 desired attitude와 total thrust를 계산하고, attitude controller를 통해 최종 torque를 결정합니다.

```text
Position Error
      ↓
Position PID
      ↓
Desired Acceleration
      ↓
Desired Attitude / Thrust
      ↓
Attitude PID
      ↓
Torque
      ↓
Rotor Mixing
```

Integral term에는 saturation을 적용하여 integral windup을 제한합니다.

---

## 3. Disturbance & Stability Analysis

### Disturbance Modeling

외란 환경을 구성하기 위해 시간에 따라 변화하는 **Wind Gust**와 **Aerodynamic Drag**를 구현했습니다.

Wind Gust는 steady wind와 stochastic gust를 포함하며, Aerodynamic Drag는 상대 속도를 기반으로 계산합니다.

$$F_{drag} = -k_{lin}|v_{rel}|v_{rel}$$

또한 rotational drag와 wind-induced torque를 고려합니다.

외란이 추가되면서 target tracking error가 증가하는 것을 확인하고 이후 외란의 영향을 직접 학습하는 방법을 적용했습니다.

### Stability Analysis

PID gain에 따른 폐루프 시스템의 안정성을 확인하기 위해 각 축을 3차 특성방정식으로 근사하여 분석합니다.

$$a_3s^3+a_2s^2+a_1s+a_0=0$$

3차 시스템에 대해 Routh-Hurwitz 조건을 적용합니다.

$$b_1=\frac{a_2a_1-a_3a_0}{a_2}$$

다음 조건을 확인하여 선형 안정성을 판별합니다.

$$a_3,a_2,a_1,a_0,b_1 > 0$$

또한 폐루프 pole을 계산하고 다음 특성을 분석합니다.

* Pole real part \(\sigma\)
* Pole imaginary part \(\omega_d\)
* Natural frequency \(\omega_n\)
* Damping ratio \(\zeta\)

$$\omega_n=\sqrt{\sigma^2+\omega_d^2}$$

$$\zeta=-\frac{\sigma}{\omega_n}$$

이를 통해 안정성뿐만 아니라 진동 성분과 감쇠 특성을 함께 확인합니다.

> 현재 안정성 분석은 축별 선형 폐루프 특성방정식에 대한 분석이며, 전체 nonlinear quadrotor system의 안정성을 직접 증명하는 것은 아닙니다.

---

## 4. Residual Dynamics Learning

외란이 존재할 때 nominal dynamics와 실제 dynamics 사이의 차이를 학습합니다.

$$a_{res}=a_{actual}-a_{nominal}$$

$$\alpha_{res}=\alpha_{actual}-\alpha_{nominal}$$

따라서 학습 대상은 다음 6차원 벡터입니다.

$$y=[a_{res,x},a_{res,y},a_{res,z},\alpha_{res,x},\alpha_{res,y},\alpha_{res,z}]$$

### Dataset

입력은 현재 12-state와 4개의 rotor thrust로 구성됩니다.

$$16D =[state_{12},T_1,T_2,T_3,T_4]$$

총 100,000개의 sample을 생성하고 80/20으로 training/validation set을 구성합니다.

`MinMaxScaler`를 이용하여 입력과 출력의 scaling을 수행합니다.

### Neural Network

Residual dynamics는 MLP를 이용하여 학습합니다.

```text
16
 ↓
128
 ↓
128
 ↓
64
 ↓
6
```

각 hidden layer에는 SiLU activation을 사용하며, MSE loss와 AdamW optimizer를 사용합니다.

Validation loss를 기준으로 best model을 저장하고, 학습 후 physical unit에서 RMSE와 MAE를 평가합니다.

---

## 5. Data-driven Compensation

학습된 residual dynamics model을 controller에 연결하여 외란 보상을 수행합니다.

```text
Current State
     +
Previous Rotor Thrust
     ↓
Residual Dynamics Model
     ↓
Predicted Residual
     ↓
Compensation
     ↓
PID Controller
     ↓
Rotor Thrust
```

예측된 linear acceleration residual은 desired acceleration에 보상합니다.

$$a_{cmd,new}=a_{cmd}-\hat a_{res}$$

Angular acceleration residual은 관성 모멘트를 이용하여 torque compensation에 반영합니다.

$$\tau_{comp}=I\hat{\alpha}_{res}$$

현재 구현에서는 이전 timestep에서 실제로 적용된 rotor thrust를 model input에 포함하여 causal한 형태로 residual prediction을 수행합니다.

### Current Status

Residual model의 학습 및 controller integration까지 구현했지만, **전체 closed-loop control 성능이 기대에 크게 못 미쳐 원인을 분석하고 있습니다.**