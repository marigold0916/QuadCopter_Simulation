# Quadrotor Simulation & Data-Driven Control

Python 기반의 쿼드콥터 시뮬레이터를 구현하고, 외란 환경에서의 제어 성능을 분석한 뒤 Data-Driven Control을 적용하는 프로젝트입니다.

## Project Overview

프로젝트는 다음과 같은 단계로 진행되었습니다.

```text
Quadrotor Dynamics
       │
       ▼
 PID Controller
       │
       ▼
Disturbance Modeling
       │
       ▼
 Stability & Error Analysis
       │
       ▼
Data-Driven Control
       │
       ▼
Residual Correction
```

### 1. PID-based Control

먼저 쿼드콥터의 6-DOF 동역학 모델과 PID 기반 위치 및 자세 제어기를 구현했습니다.

* 6-DOF rigid-body dynamics
* Position control
* Attitude control
* PID controller
* Numerical integration
* 3D simulation visualization

기본적인 제어 환경에서 목표 위치를 추종하도록 쿼드콥터를 제어합니다.

### 2. Disturbance & Stability Analysis

이후 실제 환경에서 발생할 수 있는 외란을 고려하기 위해 시뮬레이션에 외란 모델을 추가했습니다.

* Wind disturbance
* Aerodynamic drag
* Disturbance-induced tracking error

외란이 적용되면서 목표 위치와 실제 위치 사이의 오차가 증가하는 현상을 확인하고, 제어 시스템의 안정성과 응답 특성을 분석할 수 있도록 관련 계산 및 진단 기능을 추가했습니다.

### 3. Data-Driven Control

외란으로 인해 발생하는 제어 오차를 데이터 기반으로 보정하기 위해 Data-Driven Control을 시도하고 있습니다.

기존 제어기가 계산한 제어 입력과 실제 시스템의 응답 사이에서 발생하는 오차를 데이터로 수집하고, 이를 학습하여 기존 제어기를 보정하는 모델을 구성했습니다.

현재 구조는 다음과 같습니다.

```text
              ┌──────────────────┐
Target ──────►│ PID Controller   │──────► Control Input
              └──────────────────┘              │
                                                ▼
                                         ┌─────────────┐
                              Disturbance │ Quadrotor  │
                                         │  Dynamics  │
                                         └──────┬──────┘
                                                │
                                                ▼
                                           Actual State
                                                │
                                                ▼
                                      Tracking Error
                                                │
                                                ▼
                                      Residual Learning
                                                │
                                                ▼
                                      Correction Input
```

### 4. Current Status

현재 Data-Driven Control 모델은 학습 및 추론 과정까지 구현되어 있으나 제어 성능 향상으로 안정적으로 연결되지는 않고 있습니다.

현재 원인을 분석하고 있습니다.