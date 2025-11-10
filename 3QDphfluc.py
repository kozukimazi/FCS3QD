###################################
#######parallelwithphonon##########

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import eig
from scipy import integrate
from scipy.linalg import logm 
import cmath
import os
import multiprocessing as mp
from multiprocessing import Pool

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
auxdag = np.kron(sigmaz,sigmaup)
aux = np.kron(sigmaz,sigmadown)

auxd = np.kron(sigmaz,sigmaz)
#Jordan-Wigner
dldag = np.kron(sigmaup,np.eye(4))
dl = np.kron(sigmadown,np.eye(4))

drdag = np.kron(auxdag,np.eye(2))
dr = np.kron(aux,np.eye(2))

dddag = np.kron(auxd,sigmaup)
dd = np.kron(auxd,sigmadown)

nd = np.matmul(dddag,dd)
nl = np.matmul(dldag,dl)
nr = np.matmul(drdag,dr)

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

def secondd(t,x):
    N = np.shape(t)[0]
    second = []
    ts = []
    for i in range(1,N-1):
        derivada = (x[i+1] - 2*x[i] + x[i-1] )/(t[i+1] - t[i])**2
        second.append(derivada)
        ts.append(t[i])
    return ts,second

def quadrature(x1,y1):
    n = len(x1)-1
    total = 0
    for ns in range(n):
        total += (x1[ns+1] - x1[ns])*(y1[ns+1] + y1[ns])*(1/2)
    return total 

def Dl(E,U,Uf,mul,betal,gammal,gammalU):
    d = len(nl)
    auxl1 = np.sqrt( fermi(E,mul,betal)*gammal )*np.matmul( np.matmul((np.eye(d)-nr),(np.eye(d)-nd)),dldag )
    auxl2 = np.sqrt( (1-fermi(E,mul,betal))*gammal )*np.matmul( np.matmul((np.eye(d)-nr),(np.eye(d)-nd)),dl)
    auxl3 = np.sqrt( fermi(E+U,mul,betal)*gammalU )*np.matmul( np.matmul((np.eye(d)-nr) ,nd),dldag )
    auxl4 = np.sqrt( (1-fermi(E+U,mul,betal))*gammalU )*np.matmul(np.matmul((np.eye(d)-nr) ,nd),dl)
    auxl5 = np.sqrt( fermi(E+Uf,mul,betal)*gammal )*np.matmul( np.matmul((np.eye(d)-nd) ,nr),dldag )
    auxl6 = np.sqrt( (1-fermi(E+Uf,mul,betal))*gammal )*np.matmul(np.matmul((np.eye(d)-nd) ,nr),dl)
    auxl7 = np.sqrt( fermi(E+U+Uf,mul,betal)*gammalU )*np.matmul( np.matmul(nr,nd),dldag )
    auxl8 = np.sqrt( (1-fermi(E+U+Uf,mul,betal))*gammalU )*np.matmul(np.matmul(nr,nd),dl)

    return [auxl1,auxl2,auxl3,auxl4,auxl5,auxl6,auxl7,auxl8]
#operadores del disipador dr
def Dr(E,U,Uf,mur,betar,gammar,gammarU):
    d = len(nr)
    auxr1 = np.sqrt( fermi(E,mur,betar)*gammar )*np.matmul( np.matmul((np.eye(d)-nl),(np.eye(d)-nd)),drdag )
    auxr2 = np.sqrt( (1-fermi(E,mur,betar))*gammar )*np.matmul( np.matmul((np.eye(d)-nl),(np.eye(d)-nd)),dr)
    auxr3 = np.sqrt( fermi(E+U,mur,betar)*gammarU )*np.matmul( np.matmul((np.eye(d)-nl) ,nd),drdag )
    auxr4 = np.sqrt( (1-fermi(E+U,mur,betar))*gammarU )*np.matmul(np.matmul((np.eye(d)-nl) ,nd),dr)
    auxr5 = np.sqrt( fermi(E+Uf,mur,betar)*gammar )*np.matmul( np.matmul((np.eye(d)-nd) ,nl),drdag )
    auxr6 = np.sqrt( (1-fermi(E+Uf,mur,betar))*gammar )*np.matmul(np.matmul((np.eye(d)-nd) ,nl),dr)
    auxr7 = np.sqrt( fermi(E+U+Uf,mur,betar)*gammarU )*np.matmul( np.matmul( nl,nd),drdag )
    auxr8 = np.sqrt( (1-fermi(E+U+Uf,mur,betar))*gammarU )*np.matmul(np.matmul(nl ,nd),dr)

    return [auxr1,auxr2,auxr3,auxr4,auxr5,auxr6,auxr7,auxr8]

#operadores del disipador dd
def Dd(E,U,mud,betad,gammad,gammadU):
    d = len(nr)
    auxd1 = np.sqrt( fermi(E,mud,betad)*gammad )*np.matmul( np.matmul((np.eye(d)-nl),(np.eye(d)-nr)),dddag )
    auxd2 = np.sqrt( (1-fermi(E,mud,betad))*gammad )*np.matmul( np.matmul((np.eye(d)-nl),(np.eye(d)-nr)),dd)
    auxd3 = np.sqrt( fermi(E+U,mud,betad)*gammadU )*np.matmul( np.matmul((np.eye(d)-nl) ,nr) + np.matmul((np.eye(d)-nr) ,nl) ,dddag )
    auxd4 = np.sqrt( (1-fermi(E+U,mud,betad))*gammadU )*np.matmul(np.matmul((np.eye(d)-nl) ,nr) + np.matmul((np.eye(d)-nr) ,nl),dd)
    auxd5 = np.sqrt( fermi(E+ (2*U),mud,betad)*gammadU )*np.matmul( np.matmul(nr ,nl),dddag )
    auxd6 = np.sqrt( (1-fermi(E+(2*U),mud,betad))*gammadU )*np.matmul(np.matmul(nr ,nl),dd)

    return [auxd1,auxd2,auxd3,auxd4,auxd5,auxd6]

def Dissipator(E,Ed,U,Uf,mul,mur,mud,betal,betar,betad,gammal,gammalU,gammar,gammarU,gammad,gammadU):
    DR = Dr(E,U,Uf,mur,betar,gammar,gammarU)
    DL = Dl(E,U,Uf,mul,betal,gammal,gammalU)
    DD = Dd(Ed,U,mud,betad,gammad,gammadU)

    tot = []
    for l in DL:
        tot.append(l)
    for r in DR:
        tot.append(r)
    for d in DD:
        tot.append(d)    

    return tot

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

def FCS(H,Ls,Ll,chi):
    tot = Liouvillian(H,Ls) + FCSLiou(H,Ll,chi) 
    return tot

def Propagate(rho0,superop,t):
    d = len(rho0)
    propagator = expm (superop *t)
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1))
    return np.reshape( vec_rho_t, (d,d) )

#Heres the cumulants of the net number of transported electrons (we can do it with energy)
def Sl(H,Ls,Ll,rho0,chi,t):
    FC = FCS(H,Ls,Ll,chi)
    Tot = Propagate(rho0,FC,t)
    S = np.log(np.trace(Tot))
    return S

#another way calculating the largest real part of L(\chi)
def Lambdachi(H,Ls,Ll,chi):
    Lchi = FCS(H,Ls,Ll,chi)
    #print(Lchi)
    #we diagonalize L(chi)
    evals, evecs = eig(Lchi )
    reals = []
    for re in evals:
        reals.append(re.real) 
    #We choose the index of the largest real part
    n = reals.index(max(reals))
    return evals[n]
    
def Nl(H,Ls,Ll):
    #here we need to derivate around chi
    N = 10
    #here there is error
    #here we calculate the derivate of the largest real part
    chis = np.linspace(0,0.05,N)
    Ss = []
    for chi in chis:
        L = Lambdachi(H,Ls,Ll,chi)
        #print(L.imag)
        Ss.append(L)
    chisf,dS = derivada(chis,Ss)
    chisff,ddS = secondd(chis,Ss)
    return -1j*dS[0],-ddS[0]

def Nlnew(g0,params):
    E0,Ed0,U00,Uf0 = params["E0"],params["Ed0"],params["U00"],params["Uf0"]
    ev,mud0 = params["ev"],params["mud0"]
    betal,betar,betad = params["betal"],params["betar"],params["betad"]
    gl,glU = params["gl"],params["glU"]
    gr,grU = params["gr"],params["grU"]
    gd,gdU = params["gd"],params["gdU"]
    Ls0 = Dissipator(E0,Ed0,U00,Uf0,ev/2,-ev/2,mud0,betal,betar,betad,gl,glU,gr,grU,gd,gdU)
    H0 = Hamiltonian(E0,Ed0,U00,Uf0,g0)
    
    Ll0 = Dl(E0,U00,Uf0,ev/2,betal,gl,glU)
    #here we need to derivate around chi
    N = 10
    #here there is error
    #here we calculate the derivate of the largest real part
    chis = np.linspace(0,0.05,N)
    Ss = []
    for chi in chis:
        L = Lambdachi(H0,Ls0,Ll0,chi)
        #print(L.imag)
        Ss.append(L)
    chisf,dS = derivada(chis,Ss)
    chisff,ddS = secondd(chis,Ss)
    return -1j*dS[0],-ddS[0]

def Hamiltonian(E,Ed,U,Uf,g):
    a1 = E*nl + E*nr + Ed*nd
    a2 = g*( np.matmul(dldag,dr) + np.matmul(drdag,dl) )
    a3 = U* (np.matmul(nl,nd) +  np.matmul(nr,nd) ) + Uf*np.matmul(nl,nr) 
    return a1+a2+a3

################################
######here paralellize##########
################################

def run_parallel():
    # -----------------------
    # Diccionario de parámetros fijos
    # -----------------------
    params = {
        "E0": 1.0,
        "Ed0": 2-20,
        "U00": 0,
        "Uf0": 40,
        "ev": 20,
        "mud0": 2,
        "betal": 1/100,
        "betar": 1/100,
        "betad": 1/2,
        "gl": 1/100, "glU": 1/600,
        "gr": 1/600, "grU": 1/100,
        "gd": 1/50, "gdU": 1/50,
    }

    # -----------------------
    # Rango de voltajes ev a barrer
    # -----------------------
    Num = 8000
    g0s = np.linspace(0, 1, Num)

    # -----------------------
    # Paralelización con Pool
    # -----------------------
    with mp.Pool(processes=mp.cpu_count()) as pool:
        resultados = pool.starmap(Nlnew, [(g0, params) for g0 in g0s])

    # -----------------------
    # Separar los resultados en listas individuales
    # -----------------------
    Ilist = [r[0].real for r in resultados]  # -1j*dS[0]
    Slist = [r[1].real for r in resultados]  # -ddS[0]

    return g0s, Ilist, Slist


if __name__ == "__main__":
    g0s, I, S = run_parallel()
    np.savez("current_noise.npz", g0s=g0s, I=I, S=S)

    gl = 1/100
    n = len(g0s)
    gaux = []
    Il0 = []
    I2l0 = []
    for i in range(n):
        gaux.append(g0s[i]/gl)
        Il0.append(I[i]/gl)
        I2l0.append(S[i]/gl)

    # graficar o guardar
    plt.plot(gaux, Il0, label="Current")
    plt.plot(gaux, I2l0, label="Noise")
    plt.xlabel("g0/gam")
    plt.xscale("log")
    plt.legend()
    plt.show()