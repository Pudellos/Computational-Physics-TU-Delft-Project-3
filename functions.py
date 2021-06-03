import numpy as np

def outer_product(x, y):
    return x * y.T

def commutator(x, y):
    return x * y - y * x

def anti_commutator(x, y):
    return x * y + y * x

def coherent_state(alpha, N):
    X = np.matrix('0;'*(N - 1) + '0')
    for n in range(N):
        X[n] = np.exp(- np.abs(alpha)**2 / 2) * (alpha**n / np.math.factorial(n))
    return X

def solve_lindblad(H, rho0, L, gamma, timesteps, dt):
    N = rho0.shape[0]
    rho = np.ndarray(shape = (timesteps, N, N), dtype = np.complex)
    rho[0] = rho0

    for t in range(1, timesteps):

        A = np.matrix([[0, 0], [0, 0]], dtype = np.complex)

        for i in range(len(L)):
            A += gamma[i] * (L[i] * rho[t-1] * L[i].H - (1/2) * anti_commutator(L[i].H * L[i] , rho[t-1]))
    
        rho[t] = rho[t-1] + (-1j * commutator(H, rho[t-1]) + A) * dt ########### CHANGE H TO H_B TO GET STUFF IN B-FIELD ##############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    return rho