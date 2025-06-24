import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import integrate


sigmax = np.array([[0,1],
                   [1,0]])

sigmay = np.array([[0,-1j],
                   [1j,0]])

iden = np.array([[1,0],
                 [0,1]])

sigmaz = np.array([[1,0],
                   [0,-1]])

sigmaup = (sigmax + 1j*sigmay)/2
sigmadown = (sigmax - 1j*sigmay)/2

dsdag = np.kron(sigmaup,np.eye(2))
ds = np.kron(sigmadown,np.eye(2))

dddag = np.kron(sigmaz,sigmaup)
dd = np.kron(sigmaz,sigmadown)

#se cumple con Jordan Wigner
tot = np.matmul(dddag,dd) + np.matmul(dd,dddag)
#print(tot)

def fermi(E,mu,beta):
    return 1/(np.exp((E-mu)*beta) + 1)


def Dissipator(E,U,mus,mud,betas,betad,gammas,gammad):
    Ns = np.matmul(dsdag,ds)
    Nd = np.matmul(dddag,dd)
    d = len(Ns)
    auxs1 = np.sqrt( fermi(E,mus,betas)*gammas )*np.matmul( (np.eye(d)-Nd),dsdag )
    auxs2 = np.sqrt( (1-fermi(E,mus,betas))*gammas )*np.matmul( (np.eye(d)-Nd),ds)
    auxs3 = np.sqrt( fermi(E+U,mus,betas)*gammas )*np.matmul( Nd,dsdag )
    auxs4 = np.sqrt( (1-fermi(E+U,mus,betas))*gammas )*np.matmul( Nd,ds)

    
    auxd1 = np.sqrt( fermi(E,mud,betad)*gammad )*np.matmul( (np.eye(d)-Ns),dddag )
    auxd2 = np.sqrt( (1-fermi(E,mud,betad))*gammad )*np.matmul( (np.eye(d)-Ns),dd)
    auxd3 = np.sqrt( fermi(E+U,mud,betad)*gammad )*np.matmul( Ns,dddag )
    auxd4 = np.sqrt( (1-fermi(E+U,mud,betad))*gammad )*np.matmul( Ns,dd)    

    return [auxs1,auxs2,auxs3,auxs4,auxd1,auxd2,auxd3,auxd4]

def Ds(E,U,mus,betas,gammas):
    Ns = np.matmul(dsdag,ds)
    Nd = np.matmul(dddag,dd)
    d = len(Ns)
    auxs1 = np.sqrt( fermi(E,mus,betas)*gammas )*np.matmul( (np.eye(d)-Nd),dsdag )
    auxs2 = np.sqrt( (1-fermi(E,mus,betas))*gammas )*np.matmul( (np.eye(d)-Nd),ds)
    auxs3 = np.sqrt( fermi(E+U,mus,betas)*gammas )*np.matmul( Nd,dsdag )
    auxs4 = np.sqrt( (1-fermi(E+U,mus,betas))*gammas )*np.matmul( Nd,ds)

    return [auxs1,auxs2,auxs3,auxs4]

def Dd(E,U,mud,betad,gammad):
    Ns = np.matmul(dsdag,ds)
    Nd = np.matmul(dddag,dd)
    d = len(Ns)
    auxd1 = np.sqrt( fermi(E,mud,betad)*gammad )*np.matmul( (np.eye(d)-Ns),dddag )
    auxd2 = np.sqrt( (1-fermi(E,mud,betad))*gammad )*np.matmul( (np.eye(d)-Ns),dd)
    auxd3 = np.sqrt( fermi(E+U,mud,betad)*gammad )*np.matmul( Ns,dddag )
    auxd4 = np.sqrt( (1-fermi(E+U,mud,betad))*gammad )*np.matmul( Ns,dd)   

    return [auxd1,auxd2,auxd3,auxd4]


def Liouvillian( H,Ls, hbar = 1):
    d = len(H)
    superH = -1j/hbar * (np.kron(np.eye(d), H ) - np.kron(H.T,  np.eye(d))   )
    superL = sum( [np.kron(L.conjugate(),L) - 1/2 * (np.kron( np.eye(d), L.conjugate().T.dot(L)) +
                                                     np.kron( L.T.dot(L.conjugate()),np.eye(d) ))
                                                      for L in Ls ] )    
    
    return superH + superL

def FCSLiou(H,Ls,chi,hbar = 1):
    d = len(H)
    v1 = (np.exp(1j*chi)-1)
    v2 = (np.exp(-1j*chi)-1)

    n = 0
    superL = np.zeros((d**2,d**2),dtype = np.complex_)
    for L in Ls:
        if ((n%2)==0):
            superL +=  v1*np.kron(L.conjugate(),L) 
        else:    
            superL +=  v2*np.kron(L.conjugate(),L) 
        n+=1    
    return superL

def FCS(H,Ls,Lr,Ll,chi,beta):
    tot = Liouvillian(H,Ls) + FCSLiou(H,Ll,chi) + FCSLiou(H,Lr,beta)
    return tot

def Propagate(rho0,superop,t):
    d = len(rho0)
    propagator = expm (superop *t)
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1))
    return np.reshape( vec_rho_t, (d,d) )


def quadrature(x1,y1):
    n = len(x1)-1
    total = 0
    for ns in range(n):
        total += (x1[ns+1] - x1[ns])*(y1[ns+1] + y1[ns])*(1/2)
    return total  

def DistL(H,Ls,Lr,Ll,rho0,t,n):
    chis = np.linspace(-np.pi,np.pi,100)
    integrand = []
    for chi in chis:
        FC = FCS(H,Ls,Lr,Ll,chi,0)
        Tot = Propagate(rho0,FC,t)
        Prob = np.exp(-1j*chi*n)*np.trace(Tot)
        integrand.append(Prob/(2*np.pi))
    tot = quadrature(chis,integrand)    
    return tot


def Hamiltonian(eps,U,g):
    Ns = np.matmul(dsdag,ds) 
    Nd = np.matmul(dddag,dd)
    a1 = eps*( Ns + Nd)
    a2 = g*( np.matmul(dsdag,dd) + np.matmul(dddag,ds) )
    a3 = U* (np.matmul(Ns,Nd)  )
    return a1+a2+a3


U0 =0.1
g0 = 0.01

eV = 1.0
muL1 = eV/2
muR1 = -eV/2

Ls = Dissipator(0,U0,muL1,muR1,1,1,1/100,1/100)
Ll = Ds(0,U0,muL1,1,1/100)
Lr = Dd(0,U0,muR1,1,1/100)
H = Hamiltonian(0,U0,g0)

superop = Liouvillian(H,Ls)

rho0 = np.array([[1/4,0,0,0],
                 [0,1/4,0.,0],
                 [0,0.,1/4,0],
                 [0,0,0,1/4]])



# time steps
Ns = np.matmul(dsdag,ds)
Nd = np.matmul(dddag,dd)
d = len(Ns)
print(np.matmul(np.eye(d)-Nd,Nd))

#4000,200/8000,400
times = np.linspace(0,16000,1200)
Probnt1 = []
Probnt2 = []
Probnt4 = []
Probntmin = []
ProbF = []
for ts in times:
    cal1 = DistL(H,Ls,Lr,Ll,rho0,ts,1)
    Probnt1.append(cal1.real)
    cal2 = DistL(H,Ls,Lr,Ll,rho0,ts,2)
    Probnt2.append(cal2.real)
    cal4 = DistL(H,Ls,Lr,Ll,rho0,ts,-1)
    Probnt4.append(cal4.real)
    calmin = DistL(H,Ls,Lr,Ll,rho0,ts,6)
    calt = cal1 + cal2 + cal4 + calmin
    Probntmin.append(calmin.real)
    ProbF.append(calt.real)


plt.scatter(times,Probnt1,label = "n=1")
plt.scatter(times,Probnt2,label = "n=2")
plt.scatter(times,Probnt4, label = "n=-1")
#plt.scatter(times,Probntmin,label = "n=6")
#plt.scatter(times,ProbF,label = "ProbT")

plt.legend()
plt.xlabel(r'$t$',fontsize = 20)
plt.ylabel(r'$P(n,t)$',fontsize = 20)
plt.show()