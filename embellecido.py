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

def Sr(H,Ls,Lr,Ll,rho0,beta,t):
    FC = FCS(H,Ls,Lr,Ll,0,beta)
    Tot = Propagate(rho0,FC,t)
    S = np.log(np.trace(Tot))
    return S


def Nl(H,Ls,Lr,Ll,rho0,t):
    #here we need to derivate around chi
    chif = 0.001
    N = 10
    #here there is error
    h = chif/N
    chis = np.linspace(0,0.005,N)
    Ss = []
    for chi in chis:
        S = Sl(H,Ls,Lr,Ll,rho0,chi,t)
        Ss.append(S)
    chisf,dS = derivada(chis,Ss)
    chisff,ddS = derivada(chisf,dS)
    return -1j*dS[0],-ddS[0] 

def Nr(H,Ls,Lr,Ll,rho0,t):
    betaf = 0.001
    N = 10
    h = betaf/N
    betas = np.linspace(0,0.005,N)
    Ss = []
    for beta in betas:
        S = Sr(H,Ls,Lr,Ll,rho0,beta,t)
        Ss.append(S)
    betaf,dS = derivada(betas,Ss)
    betaff,ddS = derivada(betaf,dS)
    return -1j*dS[0],-ddS[0]    

def Currents(H,Ls,Lr,Ll,rho0,t):
    eps = 0.7
    tiempo = np.linspace(t-eps,t,10)
    Nls = []
    Nrs = []
    Flucl = []
    Flucr = []

    #here we have to make the list because we have to derivate
    for t in tiempo:
        Nlp,N2lp = Nl(H,Ls,Lr,Ll,rho0,t)
        #Nrp,N2rp = Nr(H,Ls,Lr,Ll,rho0,t)

        Nls.append(Nlp.real)
        #Nrs.append(Nrp.real)
        Flucl.append(N2lp.real)
        #Flucr.append(N2rp.real)

    #here we take the derivate of t
    tl,Il = derivada(tiempo,Nls)
    #tr,Ir = derivada(tiempo,Nrs)
    tl2,I2l = derivada(tiempo,Flucl)
    #tr2,I2r = derivada(tiempo,Flucr)

    #i take the last time 
    #here the derivate is a little inestable so we take the average of the last three
    return Il[-1],(I2l[-1] + I2l[-2] + I2l[-3])/3


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

def DistR(H,Ls,Lr,Ll,rho0,t,n):
    betas = np.linspace(-np.pi,np.pi,100)
    integrand = []
    for beta in betas:
        FC = FCS(H,Ls,Lr,Ll,0,beta)
        Tot = Propagate(rho0,FC,t)
        Prob = np.exp(-1j*beta*n)*np.trace(Tot)
        integrand.append(Prob/(2*np.pi))
    tot = quadrature(betas,integrand)    
    return tot


def Hamiltonian(eps,U,g):
    Ns = np.matmul(dsdag,ds) 
    Nd = np.matmul(dddag,dd)
    a1 = eps*( Ns + Nd)
    a2 = g*( np.matmul(dsdag,dd) + np.matmul(dddag,ds) )
    a3 = U* (np.matmul(Ns,Nd)  )
    return a1+a2+a3


U0 =0.
g0 = 0.005

eV = 6.5
mus1 = eV/2
mud1 = -eV/2
mul = eV/20

betas,betad,betal = 1,1,1
gs,gd,gl = 1/100,1/100,0
Ls = Dissipator(0,U0,mus1,mud1,mul,betas,betad,betal,gs,gd,gl)
Ll = Ds(0,U0,mus1,betas,gs)
Lr = Dd(0,U0,mud1,betad,gd)
H = Hamiltonian(0,U0,g0)

superop = Liouvillian(H,Ls)

rho0 = np.array([[1/4,0,0,0],
                 [0,1/4,0.1j,0],
                 [0,-0.1j,1/4,0],
                 [0,0,0,1/4]])

# time steps
Ns = np.matmul(dsdag,ds)
Nd = np.matmul(dddag,dd)
d = len(Ns)
print(np.matmul(np.eye(d)-Nd,Nd))

Num = 8000
gss = np.linspace(0.,1,Num)
gaux = []
Il = []
I2l = []
for g in gss:
    t=2000
    gau = g/gs
    H0 = Hamiltonian(0,U0,g)
    gaux.append(gau)
    Il0,I2l0 = Currents(H0,Ls,Lr,Ll,rho0,t) 
    #print(Il0/gs)  
    #print(g)
    Il.append(Il0/gs)
    I2l.append(I2l0/gs)
    #print(g)
    
plt.plot( gaux,Il)
plt.ylabel(r'$I_{L}/\gamma$',fontsize = 20)     
plt.xlabel(r'$g/\gamma$',fontsize = 20)
plt.xscale("log")
plt.show()
plt.plot( gaux,I2l)
plt.ylabel(r'$\langle \langle I^{2}_{L} \rangle \rangle/\gamma$',fontsize = 20)     
plt.xlabel(r'$g/\gamma$',fontsize = 20)
plt.xscale("log")
plt.show()

Num0 = 1000
eVs = np.linspace(70,100,Num)
evaux = []
Ilev = []
I2lev = []
for ev in eVs:
    t=2000
    g = 5/1000
    H0 = Hamiltonian(0,U0,g)
    Lsev = Dissipator(0,U0,-ev/2,mud1,ev/2,betas,betad,betal,gs,gd,gl)
    Llev = Ds(0,U0,ev/2,betas,gs)
    Lrev = Dd(0,U0,-ev/2,betad,gd)
    gaux.append(gau)
    Il0,I2l0 = Currents(H0,Lsev,Lrev,Llev,rho0,t) 
  #  print(Il0/g)  
    #print(g)
    Ilev.append(Il0/g)
    I2lev.append(I2l0/g)
    #print(g)
    
plt.plot( eVs,Ilev)
plt.ylabel(r'$I_{L}/\gamma$',fontsize = 20)     
plt.xlabel(r'$eV$',fontsize = 20)
plt.xscale("log")
plt.show()
plt.plot( eVs,I2lev)
plt.ylabel(r'$\langle \langle I^{2}_{L} \rangle \rangle/\gamma$',fontsize = 20)     
plt.xlabel(r'$eV$',fontsize = 20)
plt.xscale("log")
plt.show()



archivo = open("embellecido","w")
decimal_places = 7
total_width = 8
format_str = f"{{:.{decimal_places}f}}" 
#format_str = f"{{:{total_width}.{decimal_places}f}}"
for i in range(Num):
    archivo.write( format_str.format(gaux[i])) #guarda el grado del nodo
    #archivo.write(str(xs[i])) 
    archivo.write(" ") 
    #archivo.write(str(ys[i]))
    archivo.write( format_str.format(Il[i]))
    archivo.write(" ") 
    #archivo.write(str(ys[i]))
    archivo.write( format_str.format(I2l[i]))
    archivo.write("\n")