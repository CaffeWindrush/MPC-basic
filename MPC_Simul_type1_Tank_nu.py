"""
    MPC Simulation Platform
    MIMO Version
    2019 Jinsung Kim
"""
import numpy as np
import matplotlib.pyplot as plt
import pyecosqp
import control
import scipy.linalg as la


def GenAugSysMatrix(A, B, Cz, Np, Nu, minu, maxu, mindu=None, maxdu=None):    
    #-------------------------------------
    # for psi ----------------------------
    #-------------------------------------
    psi = np.zeros((0, A.shape[1]))
    for i in range(1,Np + 1):
        psi = np.vstack((psi, Cz * (A ** i)))
    #print(psi)

    #-------------------------------------
    # for gamma --------------------------
    #-------------------------------------
    nu = B.shape[1]
    nz = Cz.shape[0]
    gamma = np.zeros((nz,nu)) + Cz * B
    for i in range(1, Np):
        temp_gamma = (Cz * (A ** i)) * B + gamma[-nz:, :]
        gamma = np.vstack((gamma, temp_gamma))
    # print(gamma)

    #-------------------------------------
    # for theta --------------------------
    #-------------------------------------
    theta = np.zeros((nz * Np, 0))    # first row
    theta = np.hstack([theta, gamma])
    t1 = np.zeros((nz * Np, nu))
    for i in range(1, Nu):
        t1[:i*nz, :] = np.zeros_like((t1[:i*nz, :]))
        t1[i*nz:, :] = theta[:-(i*nz), :nu]
        theta = np.hstack([theta, t1])  
    # print('theta=', theta)

    return psi, gamma, theta


def GenConstMatrix(A, B, Cc, Np, Nu, x0, u0, psi_c, gamma_c, theta_c, minu, maxu, minz, maxz, mindu=None, maxdu=None):

    nu = B.shape[1]
    nc = Cc.shape[0]

    G = np.zeros((0, Nu * nu))
    h = np.zeros((0, 1))

    #-------------------------------------
    # for du constraints matrix-----------
    #-------------------------------------
    # for G matrix (LHS for du constraints)
    for i in range(0, Nu * nu):
        if maxdu is not None:
            tG = np.zeros((1, Nu * nu))
            tG[0, i] = 1
            G = np.vstack([G, tG])

        if mindu is not None:
            tG = np.zeros((1, Nu * nu))
            tG[0, i] = -1
            G = np.vstack([G, tG])
    
    for i in range(0, Nu):
        for ii in range(0, nu):
            if maxdu is not None:
                h = np.vstack([h, maxdu[ii]])
            if mindu is not None:
                h = np.vstack([h, -mindu[ii]])

    # print('G1=',G.shape,', h1=',h.shape)
    # print('G1=',G,', h1=',h)

    #-------------------------------------
    # for u constraints matrix------------
    #-------------------------------------
    # for F0
    F0 = np.zeros((0, Nu * nu))
    for i in range(0, nu):
        FmatMax = np.zeros((1, Nu * nu))
        FmatMax[0, i] = 1
        F0 = np.vstack([F0, FmatMax])

        FmatMin = np.zeros((1, Nu * nu))
        FmatMin[0, i] = -1
        F0 = np.vstack([F0, FmatMin])
    
    for i in range(1, Nu):
        F0Hrz = np.zeros((2 * nu, Nu * nu))
        for ii in range(0, i+1):
            F0Hrz[:, ii*nu:(ii+1)*nu] = F0[:2*nu, :nu]
        F0 = np.vstack([F0, F0Hrz])    

    F1 = F0[:, :nu]
    
    # print('F0_shape', F0.shape)
    # print('F0', F0)
    # print('F1_shape : ',F1.shape)
    # print('F1 : ',F1)

    # for f
    f = np.zeros((0, 1))
    for i in range(0, Nu):
        for ii in range(0, nu):
            f = np.vstack([f, maxu[ii]])
            f = np.vstack([f, -minu[ii]])

    # print('f_shape : ',f.shape)
    # print('f : ',f)

    uConRhs = -F1 * u0 + f

    F0 = F0[:, :Nu * nu]
    uConRhs = uConRhs[:, :Nu * nu]

    G = np.vstack([G, F0])
    h = np.vstack([h, uConRhs])
    # print('G=',G,', h=',h)
    # print('G2=',G.shape,', h2=',h.shape)


    #-------------------------------------
    # for state constraints matrix--------
    #-------------------------------------
    StatePred = psi_c * x0 + gamma_c * u0

    # for G
    # Initial Test Code
    zG = np.zeros((0, Np * nc))
    for i in range(0, Np * nc):
        GmatMax = np.zeros((1, Np * nc))
        GmatMax[0, i] = 1
        zG = np.vstack([zG, GmatMax])

        GminMax = np.zeros((1, Np * nc))
        GminMax[0, i] = -1
        zG = np.vstack([zG, GminMax])
    
    zg = np.zeros((0,1))
    for i in range(0, Np):
        for ii in range(0, nc):
            zg = np.vstack([zg, maxz[ii]])
            zg = np.vstack([zg, -minz[ii]])

    # print('zG=', zG)
    # print('zG= ',zG.shape)
    # print('zg=', zg)
    # print('zg= ' ,zg.shape)

    zG_Final = zG * theta_c
    zg_Final = zg - zG * StatePred

    G = np.vstack([G, zG_Final])
    h = np.vstack([h, zg_Final])
    # print('G3=',G.shape,', h3=',h.shape)

    return G, h

def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01
    # print('A=',A.shape,', B=',B.shape)
    # print('Q=',Q.shape,', R=',R.shape)
    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn
    
    return Xn


def dlqr_with_iteration(Ad, Bd, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE_with_iteration(Ad, Bd, Q, R)

    # compute the LQR gain
    K = np.matrix(la.inv(Bd.T * X * Bd + R) * (Bd.T * X * Ad))

    return K

# ---------------------------------------
# Mode Selection
mpc_mode = 1

# Process parameters
A1 = 28 # cm
A2 = 32
A3 = 28
A4 = 32

a1 = 0.071 # cm
a2 = 0.057
a3 = 0.071
a4 = 0.057

kc = 0.5 # V/cm
g = 981  # cm/s^2

v10_nmp = 3.00 # V
v20_nmp = 3.00

k1_nmp = 3.33 # cm^3/Vs
k2_nmp = 3.35 # cm^3/Vs

g1_nmp = 0.25 # ratio of allocated punmp capacity between lower and
            # upper tank
g2_nmp = 0.35


h10_nmp = 8.24441415257276
h20_nmp = 19.01629576919927
h30_nmp = 4.31462580236556
h40_nmp = 8.80652939083585


# Build state space model, minimum phase
T1_nmp = A1/a1*np.sqrt(2*h10_nmp/g)
T2_nmp = A2/a2*np.sqrt(2*h20_nmp/g)
T3_nmp = A3/a3*np.sqrt(2*h30_nmp/g)
T4_nmp = A4/a4*np.sqrt(2*h40_nmp/g)

A_nmp = np.matrix([[-1/T1_nmp, 0, A3/(A1*T3_nmp), 0],
      [0, -1/T2_nmp, 0, A4/(A2*T4_nmp)],
      [0, 0, -1/T3_nmp, 0],
      [0, 0, 0, -1/T4_nmp]])

B_nmp = np.matrix([[g1_nmp*k1_nmp/A1, 0],
      [0, g2_nmp*k2_nmp/A2],
      [0, (1-g2_nmp)*k2_nmp/A3],
      [(1-g1_nmp)*k1_nmp/A4, 0]])

C_nmp=np.matrix([[kc, 0, 0, 0], # Notice the measured signals are given in Volts!
      [0, kc, 0, 0]])

D_nmp=np.matrix([[0,0],[0,0]])

h = 2

#--------
# Constraints
# No constraints on du
# Pump capacities [0 10]V
# Level 1 [0 20]cm = [0 10]V
# Level 2 [0 20]cm
# Level 3 [0 20]cm
# Level 4 [0 20]cm
maxdu = np.matrix([10, 10]).T # limit on delta u slew rate
mindu = np.matrix([-10, -10]).T
maxu = np.matrix([10-v10_nmp, 10-v20_nmp]).T # limit absolute value of u
minu = np.matrix([-v10_nmp, -v20_nmp]).T
maxz = np.matrix([10-h10_nmp/2-0.1, 10-h20_nmp/2-0.1, 10-h30_nmp/2-0.1, 10-h40_nmp/2-0.1]).T # Limits on controlled outputs
minz = np.matrix([-h10_nmp/2, -h20_nmp/2, -h30_nmp/2, -h40_nmp/2]).T
# maxz = np.matrix([10-h10_nmp/2-0.1, 10-h20_nmp/2-0.1]).T # Limits on controlled outputs
# minz = np.matrix([-h10_nmp/2, -h20_nmp/2]).T

# # MPC parameters
Np = 30 # Prediction horizon
Nu = 10
# Hu = 10 # Horizon for varying input signal

sysc = control.ss(A_nmp,B_nmp,C_nmp,D_nmp)
sysd = control.c2d(sysc, h)

A = sysd.A
B = sysd.B
Cy = sysd.C
Cz = Cy
Cc = 0.5 * np.matrix(np.eye(4))
(nx, nu) = B.shape
ny = Cy.shape[0]
nz = Cz.shape[0]

# Initial Values
x0 = np.matrix([[0.0], [0.0], [0.0], [0.0]])
u0 = np.matrix([[0.0], [0.0]])
y0 = Cy * x0
x0_hat = np.matrix([[0.0], [0.0], [0.0], [0.0]])
# y0_hat = np.matrix([[0.0], [0.0]])

Q = np.diag([4, 1])
R = 0.01*np.diag([1, 1])
QQ = np.kron(np.eye(Np), Q)
RR = np.kron(np.eye(Nu), R)

W = np.diag([1, 1, 1, 1])
V = np.diag([0.01, 0.01])

# Set point trajectory
# 1) ref vector generation, col=state row=time : [x1, x2]
# 2) reshape it to len(ref)*ny, 1
refraw = np.vstack([np.zeros((int(60/h), 1)), 3*np.ones((int(1400/h), 1))])
refraw = np.hstack([refraw, np.zeros_like(refraw)])
# ref = refraw.reshape(len(refraw)*ny, 1)

# # Input disturbance trajectory
# d = np.hstack([np.zeros(int(600/h)), -1*np.ones(int(860/h))])
# d = np.array([d, np.zeros_like(s)])
# print('refraw=',refraw.shape)

# defintion of the time vector
t0 = 0
dt = h
tf = 0.3
reflen = int(len(refraw))
time = np.linspace(t0+h, (reflen-Np)*h, reflen-Np)
# print('time=',time.shape)

# psi, gamma, theta = GenAugSysMatrix(A, B, Cz, Np, Nu, minu, maxu, mindu, maxdu)
# psi_c, gamma_c, theta_c = GenAugSysMatrix(A, B, Cc, Np, Nu, minu, maxu, mindu, maxdu)
psi, gamma, theta = GenAugSysMatrix(A, B, Cz, Np, Nu, minu, maxu, mindu, maxdu)
psi_c, gamma_c, theta_c = GenAugSysMatrix(A, B, Cc, Np, Nu, minu, maxu, mindu, maxdu)

P = theta.T * QQ * theta + RR   # hessian term in QP

Kob = dlqr_with_iteration(A.T, Cy.T, W, V).T

X_REC = np.zeros((0, nx))
Y_REC = np.zeros((0, ny))
U_REC = np.zeros((0, nu))

X_REC = np.vstack([X_REC, x0.T])
Y_REC = np.vstack([Y_REC, y0.T])
U_REC = np.vstack((U_REC, u0.T))


x = x0
u = u0
y = y0

x_hat = x0_hat
# print('ref',ref[:N*nx, :])

# print(time.shape, uPred_REC[:,0].shape)
# x_data_plot = []
# y_data_plot = []

# print('psi=', psi.shape)
# print('gamma=', gamma.shape)
# print('theta=', theta.shape)
# print('psi_c=', psi_c.shape)
# print('gamma_c=', gamma_c.shape)
# print('theta_c=', theta_c.shape)

i = 0
# # ----- for Loop ------------------------
for t in time:
    print('i =', i)
    
    y = Cy * x

    ref = refraw[i,:]
    refHrzn = np.tile(ref, Np)
    refHrzn = refHrzn.reshape(Np * nz, 1)
    # print('refHrzn=', refHrzn.shape)

    if mpc_mode == 0:
        x_hat = x
    elif mpc_mode == 1:
        x_hat = A * x_hat + B * u + Kob * (y - Cy*x_hat)     

    err = refHrzn - (psi * x_hat + gamma * u)
    # err = ref[:N*nz, :] - (psi * x_hat + gamma * u)   # tracking error
    q = - theta.T * QQ * err     # gradient term in QP
    # print('q=', q.shape)
    G, h = GenConstMatrix(A, B, Cc, Np, Nu, x_hat, u, psi_c, gamma_c, theta_c, minu, maxu, minz, maxz, mindu, maxdu)
    # print('G=', G.shape)
    # print('h=', h.shape)

    sol = pyecosqp.ecosqp(P, q, A=G, B=h)
    du_OneCol = np.matrix(sol["x"]).T
    # print('du=',du_OneCol.shape)
    zz = psi * x_hat + gamma * u + theta * du_OneCol
    du = du_OneCol.reshape(Nu, nu)
    # print('zz=',zz)
    # print('zz=',zz.shape)

    zPred = zz.reshape(Np, ny)
    uPred = np.cumsum(du, axis = 0) + u.T
    # print('uPred=', uPred.shape)
    # print('du=',du.shape)

    z = zz[:ny, :]
    u = u + du[0, :].T
    # print('z=',z.shape)
    # print('u=',u.shape)

    # Record state and input trajectories
    Y_REC = np.vstack([Y_REC, y.T])
    U_REC = np.vstack([U_REC, u.T])

    # State Propagation by the closed-loop control input
    x = A * x + B * u

    X_REC = np.vstack([X_REC, x.T])
    i += 1

# -------------------
# print('X_REC = ', X_REC)
# print('U_REC = ', U_REC)\

# plt.rcParams["font.family"] = "Times New Roman"
# plt.figure(1)
# plt.subplot(211)
# plt.plot(time.T, Y_REC[:-1, 0], label="y1")
# plt.plot(time.T, Y_REC[:-1, 1], label="y2")
# plt.grid(True)
# plt.legend()

# plt.subplot(212)
# plt.plot(time.T, U_REC[:-1, 0], label="u")
# plt.plot(time.T, U_REC[:-1, 1], label="u")
# plt.grid(True)
# plt.legend()
# plt.show()

X_REC[:, 0] = X_REC[:, 0] + h10_nmp
X_REC[:, 1] = X_REC[:, 1] + h20_nmp
X_REC[:, 2] = X_REC[:, 2] + h30_nmp
X_REC[:, 3] = X_REC[:, 3] + h40_nmp
U_REC[:, 0] = U_REC[:, 0] + v10_nmp
U_REC[:, 1] = U_REC[:, 1] + v20_nmp
refraw[:, 0] = refraw[:, 0]/kc + h10_nmp
refraw[:, 1] = refraw[:, 1]/kc + h20_nmp


plt.rcParams["font.family"] = "Times New Roman"
plt.subplot(321)
plt.plot(time.T, X_REC[:-1, 0], label="x1")
plt.plot(time.T, refraw[:-Np, 0], label="x1")
plt.grid(True)
plt.legend()

plt.subplot(322)
plt.plot(time.T, X_REC[:-1, 1], label="x2")
plt.plot(time.T, refraw[:-Np, 1], label="x2")
plt.grid(True)
plt.legend()

plt.subplot(323)
plt.plot(time.T, X_REC[:-1, 2], label="x3")
plt.grid(True)
plt.legend()

plt.subplot(324)
plt.plot(time.T, X_REC[:-1, 3], label="x4")
plt.grid(True)
plt.legend()

plt.subplot(325)
plt.plot(time.T, U_REC[:-1, 0], label="u1")
plt.grid(True)
plt.legend()

plt.subplot(326)
plt.plot(time.T, U_REC[:-1, 1], label="u2")
plt.grid(True)
plt.legend()

plt.show()