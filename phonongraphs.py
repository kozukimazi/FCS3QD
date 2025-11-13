import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import eig
from scipy import integrate
from scipy.linalg import logm 

data = np.load("current_noiseg=5_10^{-3}.npz")
J0s, I, S = data["J0s"], data["I"], data["S"]

gl = 1/100
n = len(J0s)
betaph = 1/400
gaux = []
Il0 = []
I2l0 = []
for i in range(n):
    gaux.append(J0s[i]/(betaph*gl))
    Il0.append(I[i]/gl)
    I2l0.append(S[i]/gl)

#ev = 20
# graficar o guardar
plt.plot(gaux, Il0, lw = 3, label=r'$\langle \dot{N}_{L}\rangle/\kappa_{L}$')
plt.plot(gaux, I2l0, lw = 3,label=r'$\langle \dot{N}^{2}_{L}\rangle/\kappa_{L}$')
plt.xlabel(r'$J_{0}/(\beta_{ph}\kappa_{L})$',fontsize = 20)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
plt.legend(fontsize=15,loc = "upper right")
plt.xscale("log")
plt.show()

plt.plot(gaux, I, lw = 3, label=r'$\langle \dot{N}_{L}\rangle$')
plt.xscale("log")
plt.show()
