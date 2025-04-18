import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import integrate

def anticonmutador(A,B):
    return np.matmul(A,B) + np.matmul(B,A)

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

def seconderivada(t,x):
    N = np.shape(t)[0]
    der = []
    ts = []
    for i in range(N-2):
        derivada = (x[i+2] - 2*x[i+1] + x[i] )/(t[i+1] - t[i])**2
        der.append(derivada)
        ts.append(t[i])

    return ts,der  


def quadrature(x1,y1):
    n = len(x1)-1
    total = 0
    for ns in range(n):
        total += (x1[ns+1] - x1[ns])*(y1[ns+1] + y1[ns])*(1/2)
    return total  


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


#################################################
#########Aqui construimos los disipadores########
#################################################

#operadores del disipador dL
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

def Liouvillianfield( H,Ls,chi,beta,Ns, hbar = 1):
    d = len(H)
    superH = -1j/hbar * (np.kron(np.eye(d), H ) - np.kron(H.T,  np.eye(d))   )
    superL = np.zeros((d**2,d**2),dtype = np.complex_)
    vs1 = np.exp(1j*chi)
    vs2 = np.exp(-1j*chi)
    vd1 = np.exp(1j*beta)
    vd2 = np.exp(-1j*beta)
    n=0
    for L in Ls:
        dp = np.kron(L.conjugate(),L)
        e = 1/2*(np.kron( np.eye(d), L.conjugate().T.dot(L)) + np.kron( L.T.dot(L.conjugate()),np.eye(d)) )
        if (n<=Ns): 
            if ( (n%2)==0 ):
                superL += (vs1*dp)-e
            else: 
                superL += (vs2*dp)-e    

        else:
            if ( (n%2)==0 ):
                superL += (vd1*dp)-e
            else: 
                superL += (vd2*dp)-e 
    
        n = n+1   
    return superH + superL


def Propagate(rho0,superop,t):
    d = len(rho0)
    propagator = expm (superop *t)
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1))
    return np.reshape( vec_rho_t, (d,d) )


def Sl(H,Ls,rho0,chi,Ns,t):
    FC = Liouvillianfield(H,Ls,chi,0,Ns)
    Tot = Propagate(rho0,FC,t)
    S = np.log(np.trace(Tot))
    return S


def Nl(H,Ls,rho0,Ns,t):
    chif = 0.001
    #N=10
    N = 10
    h = chif/N
    #chi=0.005
    chis = np.linspace(0,0.1,N)
    Ss = []
    for chi in chis:
        S = Sl(H,Ls,rho0,chi,Ns,t)
        Ss.append(S)
    chisf,dS = derivada(chis,Ss)
    chisff,ddS = seconderivada(chisf,Ss)
    return -1j*dS[0],-ddS[0] 

def Currents(H,Ls,rho0,Ns,t):
    eps = 0.7
    tiempo = np.linspace(t-eps,t,10)
    Nls = []
    Nrs = []
    Flucl = []
    Flucr = []

    #here we have to make the list because we have to derivate
    for t in tiempo:
        Nlp,N2lp = Nl(H,Ls,rho0,Ns,t)
        #Nrp,N2rp = Nr(H,Ls,Lr,Ll,rho0,t)

        Nls.append(Nlp.real)
        #Nrs.append(Nrp.real)
        Flucl.append(N2lp.real)
        #Flucr.append(N2rp.real)


    tl,Il = derivada(tiempo,Nls)
    #tr,Ir = derivada(tiempo,Nrs)
    tl2,I2l = derivada(tiempo,Flucl)
    #tr2,I2r = derivada(tiempo,Flucr)

    #i take the last time 
    #here the derivate is a little inestable so we take the average of the last three
    return Il[-1],I2l[-1] #+ I2l[-2] + I2l[-3])/3



def Hamiltonian(E,Ed,U,Uf,g):
    a1 = E*nl + E*nr + Ed*nd
    a2 = g*( np.matmul(dldag,dr) + np.matmul(drdag,dl) )
    a3 = U* (np.matmul(nl,nd) +  np.matmul(nr,nd) ) + Uf*np.matmul(nl,nr) 
    return a1+a2+a3

def Htd(E,Ed,U,Uf):
    Htdl = E*nl + E*nr + Ed*nd + U*(np.matmul(nr,nd) + np.matmul(nl,nd)) + Uf*np.matmul(nr,nl) 
    return Htdl


#Parametros en que hay flujo de información apreciable
#betar,betad,betal = 1/10,10,1/10
#betar,betad,betal = 1/100,1/100,1/100
betar,betad,betal = 1/100,5/10,1/100
#datos maquina
gr,grU = (1/100)*(1/6), 1/100
gl,glU = 1/100, (1/100)*(1/6)
gd,gdU = 1/50,1/50 
#datos normales
#gr,grU = (1/100), 1/100
#gl,glU = 1/100, (1/100)
#gd,gdU = 1/50,1/50 
#gd,gdU = 1/100,1/100 
#
#gr,gd,gl = 1/100,1/300,1/100
#gr,gd,gl = 1/100,0,1/100


#revisar la base
#|1,1,1>,|1,1,0>,|1,0,1>,|1,0,0>,|0,1,1>,|0,1,0>,|0,0,1>,|0,0,0>
alp,alp2,alp3,alp4,alp5 = 0.,0.,0.0,0.0,0.
a,b,c,d = 1j*alp,1j*alp2,1j*alp3,1j*alp4

rho0 = np.array([[1/8,0,0,0,0,0,0,0],
                 [0,1/8,a,0,d,0,0,0],
                 [0,-a,1/8,0,0,0,0,0],
                 [0,0,0,1/8,0,0,b,0],
                 [0,-d,0,0,1/8,0,0,0],
                 [0,0,0,0,0,1/8,c,0],
                 [0,0,0,-b,0,-c,1/8,0],
                 [0,0,0,0,0,0,0,1/8]])

#ojo con el intervalo debido a la derivada
#eVs = np.linspace(0,200,1500)
#eVs = np.linspace(0,400,3000)
eVs = np.linspace(0,1000,2000)
Ils = []
Vars = []
E = 0
g0 = 5/1000
gaux = []
#numero de puntos
Nss = 7
for ev in eVs:
    print(ev)
    mud0 = 2
    #U00 = 3
    U00 = 40
    #mud0 = 1-U00/2
    Ed0 = mud0 -U00/2
    E0 = 4
    Uf0 = 500#10
    #Uf0 = 100
    #Uf0 = 70

    t = 40000
    Ls0 = Dissipator(E0,Ed0,U00,Uf0,ev/2,-ev/2,mud0,betal,betar,betad,gl,glU,gr,grU,gd,gdU)
    H0 = Hamiltonian(E0,Ed0,U00,Uf0,g0)
    Il0,I2l0 = Currents(H0,Ls0,rho0,Nss,t) 
    #print(Il0/gs)  
    #print(g)
    
    Ils.append(Il0/gl)
    Vars.append(I2l0/gl)
    #print(g)


plt.plot(eVs,Ils)
plt.ylabel(r'$I_{L}$',fontsize = 20)
plt.xlabel(r'eV', fontsize = 20)
#plt.xscale("log")
plt.show()


plt.plot(eVs,Vars,'o',color = 'black',markersize =0.3)
plt.ylabel(r'$\langle \Delta I^{2} \rangle$',fontsize = 20)
plt.xlabel(r'eV', fontsize = 20)
#plt.xscale("log")
plt.show()



