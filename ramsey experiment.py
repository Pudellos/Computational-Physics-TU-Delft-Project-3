import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.lib.arraypad import _slice_at_axis
from scipy.odr import *
from functions import commutator, anti_commutator, rotate, solve_lindblad, fit
from State import State
from scipy.optimize import curve_fit


t0 = time.time()
## Basis states ##
down = np.matrix([[1],
                  [0]], dtype = np.complex)
up = np.matrix([[0],
                [1]], dtype = np.complex)

phi_plus = (1 / np.sqrt(2)) * (np.kron(up, up) + np.kron(down, down) ) #state A and state B, respectively

plus  = (1 / np.sqrt(2)) * (up + down)

minus = (1 / np.sqrt(2)) * (up - down)

## Pauli/Jump operators ##
I = np.matrix([[1, 0],
               [0, 1]]) 
PauliM = np.matrix([[0, 1],
                     [0, 0]])
S_x, S_y, S_z = 1/2 * PauliX, 1/2 * PauliY, 1/2 * PauliZ ## Angular mmtm operators

## Axis
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

phi = 0.5
PauliPhi = np.matrix([[1, 0],
                      [0, np.exp(-1j * phi)]], dtype = np.complex)

timesteps = 5000
dt = 0.005

H = - 1 * S_z

t = np.arange(10, timesteps, 20)
PB = np.zeros(len(t))
L = [PauliM, PauliPhi]
k = [0.1, 0.01]
for i, N in enumerate(t):
    print("N = %s" %N, end = '\r' )
    tau = np.floor(N/2) - 1
    seq = [[0    , [np.pi/2, y_axis]], # [time of pulse, pulse = [rotation angle, rotation axis]]
           [2*tau, [-np.pi/2, y_axis]]]
    stateB = State(down, N)
    stateB.dm = solve_lindblad(H, stateB.dm[0], L, k, N, dt, pulse_sequence = seq)
    PB[i] = stateB.P(down)[-1]
    
tau = dt * (t / 2)
t = np.linspace(0, timesteps * dt, timesteps)
rho00 = np.real(stateB.dm[:, 0, 0])
rho11 = np.real(stateB.dm[:, 1, 1])
S_x_measured = stateB.calc_observable(S_x)

tPB=np.linspace(0,timesteps*dt,(len(PB)-5))
expdec = lambda x, a, b, c: a*np.exp(-x/b)+c
f = lambda x, a, b, c: a*np.exp(-x/b)*np.cos(x*c)
ftest = lambda x, a, b: 0.5*(1+np.exp(-x/a)*np.cos(x*b))
fPB = fit(tPB, PB[0:(len(PB)-5)] , ftest)
fS_x = fit(t[0:(len(S_x_measured)-10)], S_x_measured[0:(len(S_x_measured)-10)], f)
frho00 = fit(t[0:(len(rho00)-10)], rho00[0:(len(rho00)-10)] , expdec)
frho11 = fit(t[0:(len(rho11)-10)], rho11[0:(len(rho11)-10)] , expdec)



plt.figure
plt.plot(tPB, PB[0:(len(PB)-5)], label = r'$P_{down}$', markersize=2, color = 'r')
plt.xlabel(r'Time $t$')
plt.legend()
#plt.savefig("Pdown.svg")
plt.show()



plt.figure
plt.plot(t[0:(len(S_x_measured)-10)], S_x_measured[0:(len(S_x_measured)-10)], label = r'$\frac{1}{2}(\rho_{01}+\rho_{10})$')
plt.xlabel(r'Time $t$')
plt.legend()
#plt.savefig("mixed.svg")
plt.show()


plt.figure
plt.plot(t[0:(len(rho00)-10)], rho00[0:(len(rho00)-10)], label = r'$\rho_{00}$')
plt.plot(t[0:(len(rho11)-10)], rho11[0:(len(rho11)-10)], label = r'$\rho_{11}$')
plt.xlabel(r'Time $t$')
plt.legend()
#plt.savefig("updown.svg")
plt.show()


plt.figure
plt.plot(t, ftest(t, *fPB))
plt.show()

plt.figure
plt.plot(t[0:(len(S_x_measured)-10)], f(t[0:(len(S_x_measured)-10)], *fS_x))
plt.show()

t1 = time.time()
totaltime = t1-t0