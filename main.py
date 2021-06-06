import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
from functions import commutator, anti_commutator, rotate, solve_lindblad, fit
from State import State
from scipy.optimize import curve_fit

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
PauliX = np.matrix([[0, 1],
                     [1, 0]])
PauliY = np.matrix([[0 ,-1j],
                     [1j, 0]])
PauliZ = np.matrix([[1, 0],
                     [0,-1]])
PauliP = np.matrix([[0, 0],
                     [1, 0]])
PauliM = np.matrix([[0, 1],
                     [0, 0]])
S_x, S_y, S_z = 1/2 * PauliX, 1/2 * PauliY, 1/2 * PauliZ ## Angular mmtm operators

## Operators 2 spins states ##
XI = np.kron(PauliX,I) 
IX = np.kron(I,PauliX) 
YI = np.kron(PauliY,I) 
IY = np.kron(I,PauliY) 
MM = np.kron(PauliM, PauliM)
PP = np.kron(PauliP, PauliP)
MI = np.kron(PauliM, I)
IM = np.kron(I, PauliM)
PI = np.kron(PauliP, I)
IP = np.kron(I, PauliP)
ZI = np.kron(PauliZ, I)
IZ = np.kron(I, PauliZ)
ZZ = np.kron(PauliZ, PauliZ)

## Hamiltonian (spin-1/2 system, in B-field in z-direction) ##
omega_0 = 5
H = -(1/2)* omega_0 * PauliZ # -> hbar = 1

# Hamiltonian (spin-1/2 system, general B-field) ##
gamma = 1       # gyromagnetic ratio - needs proper value
B = [1 ,1, 1]   # Magnetic field
H_B = -gamma*(B[0]*S_x + B[1]*S_y + B[2]*S_z)

# Hamiltonian (2 coupled 1/2 spins )
J = 10
w1, w2 = 5, 5
H_entan = -w1 * np.kron(S_z, I) - w2 * np.kron(I, S_z) + J * np.kron(S_z, S_z) 

def main():
    timesteps = 2500
    dt = 0.01
    state = State(up, timesteps)
    L = [PauliP, PauliM, PauliZ]
    k_p, k_m, k_z = 0.1, 0.1, 0.1
    k = [k_p, k_m, k_z] 

    state.dm = solve_lindblad(H, state.dm[0], L, k, timesteps, dt)

    S_x_measured = state.calc_observable(S_x)
    P_up = state.P(up)
    fidelity = state.fidelity(state.dm[0])

    t = np.linspace(0, timesteps * dt, timesteps)
    plt.plot(t, S_x_measured, label = r'$\frac{1}{2}(\rho_{01}+\rho_{10})$')
    plt.plot(t, P_up, label = r'$P_{\uparrow}$')
    plt.plot(t, fidelity, label = r'Fidelity')
    plt.xlabel(r'Time $t$')
    plt.legend()
    plt.show()

    rho00 = np.real(state.dm[:, 0, 0])
    rho11 = np.real(state.dm[:, 1, 1])

    f = lambda x, a, b, c: a * np.exp(-b * x) + c
    rho00f = fit(t, rho00, f)
    print('gamma =',rho00f[1])
    rho11f = fit(t, rho11, f)
    
    plt.plot(t, f(t, *rho00f), 'r-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(rho00f))
    plt.plot(t, rho00, label = r'$\rho_{00}$', markersize=2)
    plt.xlabel(r'Time $t$')
    plt.title(r'$\rho_{00}$ fitted to a curve of the form: $\rho_{00}$ = ae^{-bt} + c')
    plt.legend()
    plt.show()
    
    plt.plot(t, f(t, *rho11f), 'r-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(rho11f))
    plt.plot(t, rho11, label = r'$\rho_{11}$', markersize=2)
    plt.xlabel(r'Time $t$')
    plt.title(r'$\rho_{11}$ fitted to a curve of the form: $\rho_{11}$ = ae^{-bt} + c')
    plt.legend()
    plt.show()
  
def test():   
    timesteps = 1000
    dt = 0.01
    rho_0 = up * up.H
    L = [PauliP, PauliM, PauliZ]
    k_p, k_m, k_z = 0.1, 0.1, 0.1
    k = [k_p, k_m, k_z] 

    rho = solve_lindblad(H, rho_0, L, k, timesteps, dt)
    rho_echo = solve_lindblad(H, rho_0, L, k, timesteps, dt, echo = True)
    
    S_x_measured = (1/2) * np.real(rho[:, 1, 0] + rho[:, 0, 1])
    S_x_measured_echo = (1/2) * np.real(rho_echo[:, 1, 0] + rho_echo[:, 0, 1])
 
    t = np.linspace(0, timesteps * dt, timesteps)
    plt.plot(t, S_x_measured, label = r'$\frac{1}{2}(\rho_{01}+\rho_{10})$')
    plt.plot(t, S_x_measured_echo, label = r'$\frac{1}{2}(\rho^e_{01}+\rho^e_{10})$')
    plt.xlabel(r'Time $t$')
    plt.legend()
    plt.show()

def main_entangled():
    timesteps = 2500
    dt = 0.01
    state = State(phi_plus, timesteps)

    L = [MM, PP, MI, IM, PI, IP, ZI, IZ, ZZ]
    mm, pp, mi, im, pi, ip, zi, iz, zz = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    k = [mm, pp, mi, im, pi, ip, zi, iz, zz]

    state.dm = solve_lindblad(H_entan, state.dm[0], L, k, timesteps, dt)

    P_up = state.P(np.kron(up * up.H, up * up.H))
    fidelity = state.fidelity(state.dm[0])
    S_x_measured = state.calc_observable(np.kron(S_x, S_x))
    purity = state.purity()

    t = np.linspace(0, timesteps * dt, timesteps)
    plt.plot(t, P_up, label = r'$P_{up}$', markersize=2)
    plt.plot(t, fidelity, label = r'$F$', markersize=2)
    plt.plot(t, S_x_measured, label = r'$\langle S_{x1}(t) \rangle$', markersize=2)
    plt.plot(t, purity, label = r'Tr$(\rho^2)$', markersize=2)
    plt.xlabel(r'Time $t$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if(sys.argv[1] == "spin_1/2"):
        main()
    if(sys.argv[1] == "spin_echo"):
        test()
    if(sys.argv[1] == "two_spins"):
        main_entangled()
