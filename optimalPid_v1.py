import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from control.matlab import *

#########################################################################
# simul parameter
#########################################################################
Ts = 0.01 # 10 ms
simEnd = 100
t = np.arange(0,simEnd,Ts)
cType = 2 # 1: P Type, 2: PI Type, 3: PID Type
oType = 5


# system transfer function
kp = 1.646295259739331
taup = 26.51790152411627
td = 1.75 #4.577130286540373 # delay

P0 = tf(kp, [taup, 1])
Pd_num, Pd_den = pade(td, 3) # 3/3 order
Pd = tf(Pd_num, Pd_den)
Ps = P0 * Pd
print('System Transfer Function: ' + str(Ps))

yout = []; T = []
yout, T = step(Ps, t)

# check plant response
plt.figure()
plt.plot(t, yout)
plt.grid(True)
plt.show()

#########################################################################
# Ziegler-Nichols Tuning Method
#########################################################################
Gm, Pm, Wcg, Wcp = margin(Ps)
print('Gain Margin: ' + str(Gm))
print('Phase Margin: ' + str(Pm))

pu = 2*np.pi/Wcg
ku = Gm

if cType == 1:
    kc = ku/2
    cont_num = kc
    cont_den = 1
elif cType == 2:
    kc = ku/2.2
    taui = pu/1.2
    cont_num = kc*np.array([1, 1/taui])
    cont_den = [1, 0]
elif cType == 3:
    kc = ku/1.7
    taui = pu/2
    taud = pu/8
    cont_num = kc*np.array([taud, 1, 1/taui])
    cont_den = [1, 0]

Cs = tf(cont_num, cont_den)
print('Controller Transfer Function: ' + str(Cs))

# check response for feedback
Total_Sys = minreal(feedback(Ps*Cs, 1))
print('Total Transfer Function: ' + str(Total_Sys))

yout = []; T = []
yout, T = step(Total_Sys, t)

plt.figure()
plt.plot(t, yout)
plt.grid(True)
plt.show()

dt = (t[1] - t[0]) / 2
t1 = np.arange(0, simEnd * 2, dt)
#########################################################################
# Test Feedback System
#########################################################################
def sim_model(x):

    num = []; den = []

    # initialize
    if cType == 1:
        kc = x[0]
        num = kc
        den = 1
    elif cType == 2:
        kc = x[0]
        taui = x[1]
        num = kc * np.array([1, 1/taui])
        den = [1, 0]
    elif cType == 3:
        kc = x[0]
        taui = x[1]
        taud = x[2]
        num = kc * np.array([taud, 1, 1/taui])
        den = [1, 0]

    C0 = 1.0*tf(num, den)
    Test_Sys = feedback(Ps*C0, 1)

    yout = []; T =[]
    yout, T = step(Test_Sys, t1)

    return yout, T


#########################################################################
# Optimal Tuning Method
#########################################################################
def objective(x, oType):

    yout, T = sim_model(x)
    err = []; err = 1 - yout

    # cost function
    if oType == 1: # ISE
        # convert array to matrix
        tmp = np.asmatrix(err)
        obj = tmp * tmp.T * dt
        obj = obj[0,0]
    elif oType == 2: # IAE
        obj = np.sum(np.abs(err)*dt)
    elif oType == 3: # ITSE
        tmp = np.asmatrix(T * err * dt)
        err = np.asmatrix(err)
        obj = tmp*err.T
        obj = obj[0,0]
    elif oType == 4: # ITAE
        obj = np.sum(T*np.abs(err)*dt)
    elif oType == 5: # ITAE^2
        obj = np.sum(((t1 ** 2) * np.abs(err) * dt) ** 2)

    print('Objective: ' + str(obj))

    return obj
#########################################################################


# initialize
if cType == 1:
    x0 = np.zeros(1)
    x0[0] = kc
elif cType == 2:
    x0 = np.zeros(2)
    x0[0] = kc
    x0[1] = taui
elif cType == 3:
    x0 = np.zeros(3)
    x0[0] = kc
    x0[1] = taui
    x0[2] = taud

# show final objective
print('Initial SSE Objective: ' + str(objective(x0, oType)))
print('Initial Value: ' + str(x0))

bnds = ((0.0, 15.0), (0.0, 10.0), (0.0, 3.0))
solution = minimize(objective, x0, oType, method='BFGS', bounds=bnds)
                    #,options={'gtol': 1e-9, 'disp': True})
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x, oType)))
print('Final Value: ' + str(x))

# check for final response
if cType == 2:
    kc = x[0]
    taui = x[1]
    num = kc * np.array([1, 1 / taui])
    den = [1, 0]
elif cType == 3:
    kc = x[0]
    taui = x[1]
    taud = x[2]
    num = kc * np.array([taud, 1, 1 / taui])
    den = [1, 0]

Cf = tf(num, den)
Final_Sys = feedback(Ps * Cf, 1)

yout_final = []; T_final = []
yout_final, T_final = step(Final_Sys, t)

yout_init = []; T_init = []
yout_init, T_init = step(Total_Sys, t)

plt.figure()
plt.plot(t, yout_init)
plt.plot(t, yout_final)
plt.grid(True)
plt.show()




