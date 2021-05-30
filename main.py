import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *

def outer_product(x, y):
    return x * y.T

def commutator(x, y):
    return x * y - y * x

def anti_commutator(x, y):
    return x * y + y * x

## Basis states ##
down = np.matrix([[1],
                  [0]])
up = np.matrix([[0],
                [1]])

plus  = (1 / np.sqrt(2)) * (up + down)
minus = (1 / np.sqrt(2)) * (up - down)

## Jump operators ##
sigma_x = np.matrix([[0, 1],
                     [1, 0]])
sigma_z = np.matrix([[1, 0],
                     [0,-1]])
sigma_p = np.matrix([[0, 0],
                     [1, 0]])
sigma_m = np.matrix([[0, 1],
                     [0, 0]])


### Magnetic Field present: ########################################################################
## Angular mmtm operators ## # -> hbar = 1
S_x = 1/2*np.matrix([[0, 1],
                [1, 0]]) 
S_y = 1/2*np.matrix([[0, -1j],
                [1j, 0]])
S_z = 1/2*np.matrix([[1, 0],
                [0, -1]])
gamma = 1 # gyromagnetic ratio - needs proper value

B = [1 ,1, 1]

H_B = -gamma*(B[0]*S_x + B[1]*S_y + B[2]*S_z)

####################################################################################################


## Hamiltonian (spin-1/2 system, in B-field)##
omega_0 = 5
H = -(1/2)* omega_0 * sigma_z # -> hbar = 1

timesteps = 2500
dt = 0.01
#rho_0 = outer_product(plus, plus)
rho_0 = outer_product(up, up)
k_p, k_m, k_z = 0.1, 0.1, 0.1
L = [sigma_p, sigma_m, sigma_z]
k = [k_p, k_m, k_z] 

rho = np.ndarray(shape = (timesteps, 2, 2), dtype = np.complex)
rho[0] = rho_0

P = np.zeros(timesteps, dtype = np.complex)
Fidel = np.zeros(timesteps, dtype = np.complex)
P[0] = np.trace(outer_product(up,up) * rho[0])
for t in range(1, timesteps):

    A = np.matrix([[0, 0], [0, 0]], dtype = np.complex)

    for i in range(len(L)):
        A += k[i] * (L[i] * rho[t-1] * L[i].H - (1/2) * anti_commutator(L[i].H * L[i] , rho[t-1]))
    
    rho[t] = rho[t-1] + (-1j * commutator(H, rho[t-1]) + A) * dt ########### CHANGE H TO H_B TO GET STUFF IN B-FIELD ##############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    P[t] = np.trace( outer_product(up, up) * rho[t-1] ) 
    Fidel[t] = np.trace(np.power(np.power(rho[t], 1/2) * rho_0 * np.power(rho[t], 1/2), 1/2))**2 # Calculation of the fidelity with respect to rho_0

#plt.plot(np.arange(0,timesteps), P)
#plt.show()


##### FITTING THE DECAY TIMES EQUATIONS FROM LECTURE NOTES 'LECTURE 8' ################################# 

rho00 = np.real(rho[:, 0, 0])
rho11 = np.real(rho[:, 1, 1])
rho01 = np.real(rho[:, 0, 1])
rho10 = np.real(rho[:, 1, 0])

S_x_measured = (1/2) * np.real(rho[:, 1, 0] + rho[:, 0, 1])
t = np.linspace(0, timesteps * dt, timesteps)

plt.plot(t, rho00, label = r'$\rho_{00}$', markersize=2)
plt.plot(t, rho11, label = r'$\rho_{11}$', markersize=2)
plt.plot(t, S_x_measured, label = r'$\frac{1}{2}(\rho_{01}+\rho_{10})$')
plt.xlabel(r'Time $t$')
plt.legend()
plt.show()
"""
def function(p, x):
    gamma, c = p
    return (c*(1-np.exp(-gamma*x)))

quad_model = Model(function)
x = np.arange(0,timesteps)
y = np.array(rho00)
x_err = np.full(len(x),np.finfo(np.float32).eps)
y_err = np.full(len(x),np.finfo(np.float32).eps)
data = RealData(x, y, sx=x_err, sy=y_err)
odr = ODR(data, quad_model, beta0=[0., 1.])
out = odr.run()
x_fit = np.linspace(x[0], x[-1], 1000)
y_fit = function(out.beta, x_fit)
plt.plot(x_fit, y_fit,color='blue',linewidth=2, label='data ODR fit')
plt.legend()
plt.show()
print('gamma=',out.beta[0])
######################################################################################3

plt.plot(np.arange(0,timesteps), Fidel)
plt.show()
"""