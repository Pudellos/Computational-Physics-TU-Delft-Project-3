import numpy as np
from scipy.optimize import curve_fit

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


def fit(xdata, ydata, f):
    '''
    This function fits xdata and ydata to an arbitrary function f
    '''
    popt, pcov = curve_fit(f, xdata, ydata)
    return popt

def commutator(x, y):
    return x * y - y * x

def anti_commutator(x, y):
    return x * y + y * x

def rotate(rho, angle, axis):
    nx, ny, nz = axis[0], axis[1], axis[2]
    U = I * np.cos(angle/2) - 1j * np.sin(angle/2) * (nx * PauliX + ny * PauliY + nz * PauliZ)
    U = U.round(10)
    return U * rho * U.H

def coherent_state(alpha, N):
    X = np.matrix('0;'*(N - 1) + '0')
    for n in range(N):
        X[n] = np.exp(- np.abs(alpha)**2 / 2) * (alpha**n / np.math.factorial(n))
    return X

def solve_lindblad(H, rho0, L, gamma, timesteps, dt, echo = False):
    N = rho0.shape[0]
    rho = np.ndarray(shape = (timesteps, N, N), dtype = np.complex)
    rho[0] = rho0

    if(not echo):
        for t in range(1, timesteps):
            A = np.matrix((('0 '*(N - 1) + '0; ') * (N - 1) + '0 '*(N - 1) + '0'), dtype = np.complex)
            for i in range(len(L)):
                A += gamma[i] * (L[i] * rho[t-1] * L[i].H - (1/2) * anti_commutator(L[i].H * L[i] , rho[t-1]))
            rho[t] = rho[t-1] + (-1j * commutator(H, rho[t-1]) + A) * dt

    if(echo):
        tau = np.floor(timesteps / 2)
        rho[0] = rotate(rho[0], np.pi/2, [0, 1, 0])
        print("done1")
        for t in range(1, timesteps):
            #print(t, timesteps)
            if( t == tau ):
                rho[t] = rotate(rho[0], np.pi, [0, 0, 1])
                print("done2")
            else:
                A = np.matrix([[0, 0], [0, 0]], dtype = np.complex)
                for i in range(len(L)):
                    A += gamma[i] * (L[i] * rho[t-1] * L[i].H - (1/2) * anti_commutator(L[i].H * L[i] , rho[t-1]))
                rho[t] = rho[t-1] + (-1j * commutator(H, rho[t-1]) + A) * dt 
        rho[t] = rotate(rho[t], np.pi/2, [0, 1, 0])
        print("done3")
    return rho

def main():
    down = np.matrix([[1],
                  [0]])
    up = np.matrix([[0],
                    [1]])
    
    rho = up * up.H

    rho = rotate(rho, np.pi/2, [0, 1, 0])
    print(rho)

    X = np.matrix([[1e-17, 0],
                   [0, 0]], dtype=np.complex)


if __name__ == "__main__":
    main()