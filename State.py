import numpy as np
import scipy.linalg 

class State:

    def __init__(self, phi0, timesteps):
        self.phi0 = phi0
        self.timesteps = timesteps
        self.N = phi0.size
        self.dm = np.ndarray(shape = (self.timesteps, self.N, self.N), dtype = np.complex)
        self.dm[0] = phi0 * phi0.H
        
    def calc_observable(self, A):
        '''
        Calculates the expectation value of observable A over time
        '''
        x = np.zeros(self.timesteps, dtype = np.complex)
        for t in range(self.timesteps):
            x[t] = np.trace( A * np.matrix(self.dm[t]) )
        return np.real(x)
    
    def P(self, m):
        '''
        Calculates the probability to measure state m
        '''
        M = m * m.H
        x = np.zeros(self.timesteps, dtype = np.complex)
        for t in range(self.timesteps):
            x[t] = np.trace( M.H * M * np.matrix(self.dm[t]) )
        return np.real(x)

    def fidelity(self, other):
        '''
        Calculates the fidelity of self w.r.t other
        '''
        sig = other
        sq_sig = scipy.linalg.sqrtm(sig).round(10)
        F = np.zeros(self.timesteps, dtype = np.complex)
        for t in range(self.timesteps):
            rho = np.matrix(self.dm[t])
            x = np.matrix(sq_sig * rho * sq_sig)
            F[t] = (np.trace( scipy.linalg.sqrtm(x)))**2
        return np.real(F)

    def purity(self):
        '''
        Calculates the purity of self (Tr(rho^2))
        '''
        p = np.zeros(self.timesteps, dtype = np.complex)
        for t in range(self.timesteps):
            rho = np.matrix(self.dm[t])
            p[t] = np.trace( rho * rho )
        return np.real(p)
        