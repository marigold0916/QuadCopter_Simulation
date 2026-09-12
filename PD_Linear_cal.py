import numpy as np
import math as m
from Controller import Controller
from QuadCopter import QuadCopter

# Kp_pos = list(map(float, input("Kp_pos 입력: ").split()))
# Kd_pos = list(map(float, input("Kd_pos 입력: ").split()))
# Kp_att = list(map(float, input("Kp_att 입력: ").split()))
# Kd_att = list(map(float, input("Kd_att 입력: ").split()))
Quad_I = QuadCopter()
Kpos = Controller()
Katt = Controller()

Inn = Quad_I.I
Kp_pos = Kpos.Kp_pos
Kd_pos = Kpos.Kd_pos
Kp_att = Katt.Kp_att
Kd_att = Katt.Kd_att

Wn_pos = np.sqrt(Kp_pos)
ksi_pos = Kd_pos * (1 / (2 * np.sqrt(Kp_pos)))

Wn_att = np.sqrt(Kp_att/Inn)
ksi_att =  Kd_att * (1/(2*np.sqrt(Kp_att*Inn)))

#Linear Stability
def Laplace_transform_att(Wn_att, ksi_att):
    #s**2 +2*ksi_att*Wn_att*s + Wn_att**2 = 0
    a = 1.0
    b = 2*ksi_att*Wn_att
    c = Wn_att**2

    s1 = (-b + np.sqrt((b**2) - 4 * a * c +0j))
    s1 = s1/(2*a)

    s2 = (-b - np.sqrt((b**2) - 4 * a * c +0j))
    s2 = s2/(2*a)
    print(f"att s1: {s1}")
    print(f"att s2: {s2}")
    return s1, s2

def Laplace_transform_pos():
        #s^2 + Kd*s +Kp = 0
    a_p = 1
    b_p = Kd_pos
    c_p = Kp_pos
    s1_pos = (-b_p + np.sqrt((b_p**2) - 4 * a_p * c_p +0j))
    s1_pos = s1_pos/(2*a_p)

    s2_pos = (-b_p - np.sqrt((b_p**2) - 4 * a_p * c_p +0j))
    s2_pos = s2_pos/(2*a_p)

    print(f"pos s1: {s1_pos}")
    print(f"pos s2: {s2_pos}")
    return s1_pos, s2_pos

def overshoot(ksi_pos, ksi_att):
    M_pos = np.zeros(len(ksi_pos))
    M_att = np.zeros(len(ksi_att))

    for i in range(len(ksi_pos)):
        if ksi_pos[i] < 1:
            M_pos[i] = np.exp(-((np.pi*ksi_pos[i]) / (np.sqrt(1-ksi_pos[i]**2))))
            M_pos[i] = M_pos[i] * 100
            print(f"pos overshoot: {M_pos[i]:.2f}%")
                
        elif ksi_pos[i] == 1:
            print(f"{ksi_pos[i]:.2f}, pos{[i+1]}임계감쇠")

        else:
            print(f"{ksi_pos[i]:.2f},  pos{[i+1]}과감쇠")

    for j in range(len(ksi_att)):        
        if ksi_att[j] < 1:
            M_att[j] = np.exp(-((np.pi*ksi_att[j]) / (np.sqrt(1-ksi_att[j]**2))))
            M_att[j] = M_att[j] * 100
            print(f"att overshoot: {M_att[j]:.2f}%")
                
        elif ksi_att[j] == 1:
            print(f"{ksi_att[j]:.2f}, att{[j+1]} 임계감쇠")
            
        else:
            print(f"{ksi_att[j]:.2f}, att{[j+1]}과감쇠")
            
    return M_pos,M_att
# def Laplace_transform_rev_att(s1,s2,e0, edot0):
#     C1 = 

att_s1, att_s2 = Laplace_transform_att(Wn_att, ksi_att)
pos_s1, pos_s2 = Laplace_transform_pos()
M_pos,M_att = overshoot(ksi_pos, ksi_att)