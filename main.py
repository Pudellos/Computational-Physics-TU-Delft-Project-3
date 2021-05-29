import numpy as np
import matplotlib.pyplot as plt

def outer_product(x, y):
    return np.matmul(x, np.transpose(y))

def commutator(x, y):
    return np.matmul(x,y) - np.matmul(y,x)

def anti_commutator(x, y):
    return np.matmul(x,y) + np.matmul(y,x)

## Basis states ##
down = np.matrix([[1],
                  [0]])
up = np.matrix([[0],
                [1]])

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
omega_0 = 1
H = -(1/2)* omega_0 * sigma_z # -> hbar = 1

timesteps = 1000
dt = 0.01
rho_0 = outer_product(up, up)
k_p, k_m, k_z = 1, 1, 1
L = [sigma_p, sigma_m, sigma_z]
k = [k_p, k_m, k_z] 

rho = np.ndarray(shape = (timesteps, 2, 2), dtype = np.complex)
rho[0] = rho_0


P = np.zeros(timesteps)
Fidel = np.zeros(timesteps)
P[0] = np.trace(np.matmul(outer_product(up,up), rho[0]))
for t in range(1, timesteps):

    A = np.matrix([[0, 0], [0, 0]], dtype = np.complex)

    for i in range(len(L)):
        A += k[i] * (np.matmul(np.matmul(L[i], rho[t-1]), L[i].getH()) - (1/2) * anti_commutator(np.matmul(L[i].getH(), L[i]), rho[t-1]))
    
    rho[t] = rho[t-1] + (-1j * commutator(H, rho[t-1]) + A) * dt ########### CHANGE H TO H_B TO GET STUFF IN B-FIELD ##############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#     print(rho[t-1])  
#     print(round(np.trace(rho[t-1]),2))
    P[t] = np.trace(np.matmul(outer_product(up,up), rho[t-1]))
    Fidel[t]=(np.trace(np.power(np.power(rho[t], 1/2)*rho_0*np.power(rho[t], 1/2), 1/2)))**2 #Calculation of the fidelity with respect to rho_0


plt.plot(np.arange(0,timesteps), P)
plt.show()


##### FITTING THE DECAY TIMES EQUATIONS FROM LECTURE NOTES 'LECTURE 8' ################################# 

rho11=np.zeros(len(rho))
for i in range(len(rho)):
    rho11[i]=rho[i][0][0]

plt.plot(np.arange(0,timesteps), rho11,'ro',label='data',markersize=2)
plt.xlabel('timesteps')
plt.ylabel('rho_11')

from scipy.odr import *
def function(p, x):
    gamma, c = p
    return (c*(1-np.exp(-gamma*x)))
quad_model = Model(function)
x = np.arange(0,timesteps)
y = np.array(rho11)
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