import numpy as np
import math as m
from Controller import Controller
from QuadCopter import QuadCopter

Quad_I =   QuadCopter()
Kpos   =   Controller()
Katt   =   Controller()

Inn    = Quad_I.I
Kp_pos = Kpos.Kp_pos
Kd_pos = Kpos.Kd_pos
Ki_pos = Kpos.Ki_pos
Ki_att = Kpos.Ki_att
Kp_att = Katt.Kp_att
Kd_att = Katt.Kd_att

s3_acc = np.ones_like(Inn) 



#stable?
def att_stability():
    att3 = Inn
    att2 = Kd_att
    att1 = Kp_att
    att0 = Ki_att
    btt1 = ((att2*att1) - (att3*att0))/att2

    att_stable = np.min([att3,att2,att1,att0,btt1])

    if (att_stable > 0).all():
        print("Att Linear Stable")
        return True
    else:
        print("Att Linear Unstable")
        print(f"att3:{att3},att2:{att2},att1:{att1},att0:{att0},btt1:{btt1}")
        return False

def acc_stability():
    acc3 = s3_acc
    acc2 = Kd_pos
    acc1 = Kp_pos
    acc0 = Ki_pos
    bcc1 = ((acc2*acc1) - (acc3*acc0))/acc2

    acc_stable = (
        (acc3 > 0) &
        (acc2 > 0) &
        (acc1 > 0) &
        (acc0 > 0) &
        (bcc1 > 0)
    )
    if np.all(acc_stable):
        print("Acc Linear Stable")
        return True
    else:
        print("Acc Linear Unstable")
        print(f"acc3:{acc3},acc2:{acc2},acc1:{acc1},acc0:{acc0},bcc1:{bcc1}")
        return False

def overshoot():
    sigma_att_list, wd_att_list = [], []
    sigma_acc_list, wd_acc_list = [], []
    for i in range(len(Inn)):
        co_att = [Inn[i],    Kd_att[i], Kp_att[i], Ki_att[i]]
        co_acc = [s3_acc[i], Kd_pos[i], Kp_pos[i], Ki_pos[i]]

        root_att = np.roots(co_att)
        root_acc = np.roots(co_acc)

        

        sigma_att_list.append(root_att.real)
        sigma_acc_list.append(root_acc.real)

        wd_att_list.append(root_att.imag)
        wd_acc_list.append(root_acc.imag)

    sigma_att = np.array(sigma_att_list)#자세 실수부
    sigma_acc = np.array(sigma_acc_list)#가속도 실수부
    wd_att    = np.array(wd_att_list)   #자세 허수부
    wd_acc    = np.array(wd_acc_list)   #가속도 허수부
    Wn_att    = np.sqrt((sigma_att**2)+(wd_att**2))
    Wn_acc    = np.sqrt((sigma_acc**2)+(wd_acc**2))
    ksi_att   = -sigma_att / Wn_att
    ksi_acc   = -sigma_acc / Wn_acc 
    # Mp_att    = np.exp(-((np.pi*ksi_att)/(np.sqrt(1-ksi_att**2))))
    # Mp_acc    = np.exp(-((np.pi*ksi_acc)/(np.sqrt(1-ksi_acc**2))))
    # Mp_att    = Mp_att * 100
    # Mp_acc    = Mp_acc * 100
    # damp_att  = len(set(map(tuple,sigma_att_list)))
    # damp_acc  = len(set(map(tuple,sigma_acc_list)))

    # if np.all(wd_att == 0) and damp_att==1:
    #     print("자세 임계감쇠")
    # elif np.all(wd_att == 0) and damp_att>1:
    #     print("자세 과감쇠")
    # elif np.any(wd_att):
    #     print(f"자세 오버슛: {Mp_att}")

    # if np.all(wd_acc == 0) and damp_acc==1:
    #     print("가속도 임계감쇠")
    # elif np.all(wd_acc == 0) and damp_acc>1:
    #     print("가속도 과감쇠")
    # elif np.any(wd_acc):
    #     print(f"가속도 오버슛: {Mp_acc}%")

    

    if np.all(sigma_att < 0):
        print("Sigma_Att is lower than 0. Att Stable")
    else:
        print("Sigma_Att is higher than 0. Att Unstable")

    if np.all(sigma_acc < 0):
        print("Sigma_Acc is lower than 0. Acc Stable")
    else:
        print("Sigma_Acc is higher than 0. Acc Unstable")

    if np.any(wd_att!=0):
        print("자세진동발생")
    else:
        print("자세진동없음")

    if np.any(wd_acc!=0):
        print("가속도진동발생")
    else:
        print("가속도진동없음")


    


acc_linear_stable = acc_stability()
att_linear_stable = att_stability()
asdf = overshoot()
