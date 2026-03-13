import numpy as np
import matplotlib.pyplot as plt

S0 = 100
T = 1
r = 0.03
mu = [-0.2,0,0.2]
sigma = [0.1,0.3,0.5]
nu = [0.2,0.4,0.6]
sims = 20000
K = np.arange(90,121,1)
dt = 1/(24*365)



N = T/dt
t = np.arange(0,T,dt)



def Bt(gam: list):
    Z = np.random.normal(0,1,int(N))
    dgamt = np.diff(gam,prepend=0)
    return np.cumsum(Z*np.sqrt(dgamt))

def gammat(k: int):
    gam = np.random.gamma(dt/nu[k],nu[k],int(N))
    return np.cumsum(gam)

def Wgamt(i: int, j: int, k: int):
    gam = gammat(k)
    return mu[i]*gam + sigma[j]*Bt(gam)

def w(i: int,j: int, k: int):
    return np.log(1-mu[i]*nu[k]-sigma[j]**2*nu[k]/2)/nu[k]

def St(i: int, j: int, k: int):
    return S0*np.exp((r+w(i,j,k))*t+Wgamt(i,j,k))

#plt.plot(t,B)
#plt.plot(t,W)
#plt.plot(t,St(1,1,1))
#plt.plot(t,gammat(0))
plt.show()