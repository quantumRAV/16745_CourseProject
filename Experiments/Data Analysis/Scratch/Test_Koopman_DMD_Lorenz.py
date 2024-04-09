
import numpy as np
from scipy.integrate import solve_ivp
from collections import namedtuple
from matplotlib import pyplot as plt

import pykoopman as pk
from pydmd import DMD
from pydmd.plotter import plot_eigs, plot_summary

def Lorenz(t,vars,params):
    x,y,z = vars
    xp = params.sigma*(y-x)
    yp = x*(params.rho - z) - y
    zp = x*y - params.beta*z
    dvars_dt = [xp,yp,zp]
    return(dvars_dt)

def DMD_man(X,Xprime,r):
    U, Sigma, VT = np.linalg.svd(X,full_matrices = False)
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T).T).T
    Lambda, W = np.linalg.eig(Atilde)
    Lambda = np.diag(Lambda)
    Phi = Xprime @ np.linalg.solve(Sigmar.T, VTr).T @ W
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda, alpha1)
    return Atilde, Ur, Phi, Lambda, b



params = namedtuple('params',['sigma','rho','beta'])
p = params(sigma = 10, rho = 28, beta = 8/3)


x0 = (0,1,20)
dt = 0.001
t = np.arange(0,50+dt,dt)
result = solve_ivp(lambda t,x: Lorenz(t,x,p),[0,50], x0,
                   t_eval=t)

x,y,z = result.y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(x, y, z,linewidth=1)
plt.show()


## ---- Compute DMD: ---- ##
X_all = np.array([x,y,z]) #should be dimension 3 x 50001, i.e. number of rows is the state variables. Number of columns is the number of observations
X = X_all[:,0:-1]
X_prime = X_all[:,1:] #time shifted data matrices

n = np.size(t)

Atilde, Ur, Phi, Lambda, b = DMD_man(X,X_prime,3)

sim_data = np.array([np.zeros(n), np.zeros(n),np.zeros(n)])
sim_data[:,0] = x0

for i in range(0,n-1):
    sim_data[:,i+1] = Ur@Atilde@Ur.conj().T@sim_data[:,i] #forward simulate -> this isnt correct
    #sim_data[:,i+1] = np.linalg.solve(Phi, Phi @ Lambda ) @ sim_data[:,i]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(sim_data[0,:], sim_data[1,:], sim_data[2,:],linewidth=1)
plt.show()

# #Koopman approach -> DMD -> compare against results above
dmd = DMD(svd_rank = 3)
model = pk.Koopman(regressor = dmd)
model.fit(X_all.T, dt = dt)
dmd.fit(X_all)
#
#
#Pure DMD results
sim_data_DMD = np.array([np.zeros(n), np.zeros(n),np.zeros(n)])
sim_data_DMD[:,0] = x0
for i in range(0, n - 1):
    sim_data_DMD[:, i + 1] = dmd.predict(sim_data_DMD[:, i]).T  # forward simulate

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(sim_data_DMD[0, :], sim_data_DMD[1, :], sim_data_DMD[2, :], linewidth=1)
plt.legend()
plt.show()

plot_summary(dmd)


#Koopman with Sparse dynamics

# Koopman results
f_predicted = np.vstack((X_all[:,0], model.simulate(X_all[:,0], n_steps=n - 1))).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(sim_data[0,:], sim_data[1,:], sim_data[2,:],linewidth=1)
plt.plot(f_predicted[0,:], f_predicted[1,:], f_predicted[2,:],linewidth=1)
plt.plot(x, y, z,linewidth=1)
plt.show()

