import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import integrate
from scipy.linalg import logm 
from scipy import integrate
import cmath
import os

#here we better the codes i had of FCS
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import integrate
from scipy.linalg import logm 
from scipy import integrate
import cmath
import os

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

#Jordan-Wigner
dsdag = np.kron(sigmaup,np.eye(2))
ds = np.kron(sigmadown,np.eye(2))

dddag = np.kron(sigmaz,sigmaup)
dd = np.kron(sigmaz,sigmadown)

#se cumple con Jordan Wigner
tot = np.matmul(dddag,dd) + np.matmul(dd,dddag)
#print(tot)

def fermi(E,mu,beta):
    return 1/(np.exp((E-mu)*beta) + 1)

def derivada(t,x):
    N = np.shape(t)[0]
    der = []
    ts = []
    for i in range(N-1):
        derivada = (x[i+1] - x[i] )/(t[i+1] - t[i])
        der.append(derivada)
        ts.append(t[i])

    return ts,der  

def quadrature(x1,y1):
    n = len(x1)-1
    total = 0
    for ns in range(n):
        total += (x1[ns+1] - x1[ns])*(y1[ns+1] + y1[ns])*(1/2)
    return total 

def Dissipator(E,U,mus,mud,mul,betas,betad,betal,gammas,gammad,gammal):
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

    auxl1 = np.sqrt( fermi(E,mul,betal)*gammal )*np.matmul( (np.eye(d)-Nd),dsdag )
    auxl2 = np.sqrt( (1-fermi(E,mul,betal))*gammal )*np.matmul( (np.eye(d)-Nd),ds)
    auxl3 = np.sqrt( fermi(E+U,mul,betal)*gammal )*np.matmul( Nd,dsdag )
    auxl4 = np.sqrt( (1-fermi(E+U,mul,betal))*gammal )*np.matmul( Nd,ds)


    return [auxs1,auxs2,auxs3,auxs4,auxd1,auxd2,auxd3,auxd4,auxl1,auxl2,auxl3,auxl4]

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

#here we add the FCS contribution
#Like L[\rho] + (e^{i\chi}-1)*(jumpterms)
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

#Heres the cumulants of the net number of transported electrons (we can do it with energy)
def Sl(H,Ls,Lr,Ll,rho0,chi,t):
    FC = FCS(H,Ls,Lr,Ll,chi,0)
    Tot = Propagate(rho0,FC,t)
    S = np.log(np.trace(Tot))
    return S


def Hamiltonian(eps,U,g):
    Ns = np.matmul(dsdag,ds) 
    Nd = np.matmul(dddag,dd)
    a1 = eps*( Ns + Nd)
    a2 = g*( np.matmul(dsdag,dd) + np.matmul(dddag,ds) )
    a3 = U* (np.matmul(Ns,Nd)  )
    return a1+a2+a3


