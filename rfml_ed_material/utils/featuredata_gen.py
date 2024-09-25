import numpy as np


class featuredata_gen:
    def __init__(self, signal_data):
        self.signal_data = signal_data

        self.N_s = len(self.signal_data)

        self.a = np.abs(self.signal_data)
        self.m_a = np.mean(self.a)
        self.a_n = self.a/self.m_a

        self.phi = np.unwrap(np.angle(self.signal_data))
        self.f = np.concatenate([[0], np.diff(self.phi)])
        self.m_f = np.mean(self.f)

        self.S = np.abs(np.fft.fftshift(np.fft.fft(self.signal_data)))

    def sigma_aa(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += self.a_n[k]**2.0
            term_2 += np.abs(self.a_n[k])
        term_1 = (1.0/self.N_s)*term_1
        term_2 = ((1.0/self.N_s)*term_2)**2.0

        return np.sqrt(term_1-term_2)

    def sigma_dp(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += self.phi[k]**2.0
            term_2 += self.phi[k]
        term_1 = (1.0/self.N_s)*term_1
        term_2 = ((1.0/self.N_s)*term_2)**2.0

        return np.sqrt(term_1-term_2)

    def sigma_ap(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += self.phi[k]**2.0
            term_2 += np.abs(self.phi[k])
        term_1 = (1.0/self.N_s)*term_1
        term_2 = ((1.0/self.N_s)*term_2)**2.0

        return np.sqrt(term_1-term_2)

    def sigma_af(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += self.f[k]**2.0
            term_2 += np.abs(self.f[k])
        term_1 = (1.0/self.N_s)*term_1
        term_2 = ((1.0/self.N_s)*term_2)**2.0

        return np.sqrt(term_1-term_2)

    def kurtosis_a(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += (self.a[k]-self.m_a)**4.0
            term_2 += (self.a[k]-self.m_a)**2.0

        return term_1/(term_2**2.0)

    def kurtosis_f(self):
        term_1 = 0.0
        term_2 = 0.0
        for k in range(self.N_s):
            term_1 += (self.f[k]-self.m_f)**4.0
            term_2 += (self.f[k]-self.m_f)**2.0

        return term_1/(term_2**2.0)

    def symmetry(self):
        P_L = 0.0
        P_U = 0.0
        for k in range(int(self.N_s/2)):
            P_L += self.S[k]**2.0
            P_U += self.S[-k]**2.0

        return (P_L-P_U)/(P_L+P_U)
