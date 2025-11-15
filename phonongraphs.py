import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import eig
from scipy import integrate
from scipy.linalg import logm 

data0 = np.load("current_noiseg=10^{-3}.npz")
J0s, I0, S0 = data0["J0s"], data0["I"], data0["S"]

data1 = np.load("current_noiseg=3_10^{-3}.npz")
J1s, I1, S1 = data1["J0s"], data1["I"], data1["S"]

data2 = np.load("current_noiseg=5_10^{-3}.npz")
J2s, I2, S2 = data2["J0s"], data2["I"], data2["S"]

#data3 = np.load("current_noiseg=7_10^{-3}.npz")
#J3s, I3, S3 = data3["evs"], data3["I"], data3["S"]

data4 = np.load("current_noiseg=10^{-4}.npz")
J4s, I4, S4 = data4["J0s"], data4["I"], data4["S"]

data5 = np.load("current_noiseg=5_10^{-4}.npz")
J5s, I5, S5 = data5["J0s"], data5["I"], data5["S"]

data6 = np.load("current_noiseg=10^{-2}.npz")
J6s, I6, S6 = data6["J0s"], data6["I"], data6["S"]

gl = 1/100
n = len(J0s)
betaph = 1/400
gaux = []
Il0 = []
I2l0 = []
Il1 = []
I2l1 = []
Il2 = []
I2l2 = []
Il3 = []
I2l3 = []
Il4 = []
I2l4 = []
Il5 = []
I2l5 = []
Il6 = []
I2l6 = []
for i in range(n):
    gaux.append(J0s[i]/(betaph*gl))
    Il0.append(I0[i]/gl)
    I2l0.append(S0[i]/gl)
    Il1.append(I1[i]/gl)
    I2l1.append(S1[i]/gl)
    Il2.append(I2[i]/gl)
    I2l2.append(S2[i]/gl)
    #Il3.append(I3[i]/gl)
    #I2l3.append(S3[i]/gl)
    Il4.append(I4[i]/gl)
    I2l4.append(S4[i]/gl)
    Il5.append(I5[i]/gl)
    I2l5.append(S5[i]/gl)
    Il6.append(I6[i]/gl)
    I2l6.append(S6[i]/gl)



#ev = 20
# graficar o guardar
plt.plot(gaux, Il0, lw = 3, label=r'$g = 10^{-3}$')
plt.plot(gaux, Il1, lw = 3,label=r'$g= 3 \times 10^{-3}$')
plt.plot(gaux, Il2, lw = 3,label=r'$g = 5 \times 10^{-3}$')
#plt.plot(gaux, Il3, lw = 3,label=r'$g = 7 \times 10^{-3}$')
plt.plot(gaux, Il4, lw = 3,label=r'$g = 10^{-4}$')
plt.plot(gaux, Il5, lw = 3,label=r'$g = 5 \times 10^{-4}$')
plt.plot(gaux, Il6, lw = 3,label=r'$g = 10^{-2}$')
plt.xlabel(r'$J_{0}/(\beta_{ph}\kappa_{L})$',fontsize = 20)
plt.ylabel(r'$\langle I\rangle/\kappa_{L}$',fontsize = 20)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
plt.legend(fontsize=15,loc = "upper right")
plt.xscale("log")
plt.show()


plt.plot(gaux, I2l0, lw = 3, label=r'$g = 10^{-3}$')
plt.plot(gaux, I2l1, lw = 3,label=r'$g= 3 \times 10^{-3}$')
plt.plot(gaux, I2l2, lw = 3,label=r'$g = 5 \times 10^{-3}$')
#plt.plot(gaux, I2l3, lw = 3,label=r'$g = 7 \times 10^{-3}$')
plt.plot(gaux, I2l4, lw = 3,label=r'$g = 10^{-4}$')
plt.plot(gaux, I2l5, lw = 3,label=r'$g = 5 \times 10^{-4}$')
plt.plot(gaux, I2l6, lw = 3,label=r'$g = 10^{-2}$')
plt.xlabel(r'$J_{0}/(\beta_{ph}\kappa_{L})$',fontsize = 20)
plt.ylabel(r'$\langle I^{2} \rangle/\kappa_{L}$',fontsize = 20)
plt.xticks(fontsize=17)  # X-axis tick labels
plt.yticks(fontsize=17)
plt.legend(fontsize=15,loc = "upper right")
plt.xscale("log")
plt.show()