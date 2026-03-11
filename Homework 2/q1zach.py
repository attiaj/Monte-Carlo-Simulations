import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 3
sigma = 0.25
rbar = 0.05
rnaught = 0.05
snaught = 100
mu = 0.15
Ns = [1,5,10,50,100,500]#,1000]
N = max(Ns)
T = 1

#t = np.linspace(0,T,N)

# Expected value of a random variable
def expectation(sims: int, expectee, N, B):
    trials = []
    i = 0
    while i < sims:
        trials.append(expectee(N,B[i]))
        i += 1
    
    return np.average(trials,axis=0)

# Browniam motion seed
def Bt(N: int):
    Z = np.random.normal(0,1,N)
    dt = np.ones(N) * T/N
    return np.cumsum(Z*np.sqrt(dt))

# rt exact simulation
def rt(N: int, B: list):
    integral = []
    j = 0
    while j < N:
        nextInt = -B[0]
        i = 1
        while i <= j:
            nextInt += np.exp(-alpha*(t[j]-t[i]))*(B[i]-B[i-1])
            i += 1
        integral.append(nextInt)
        j += 1
    return rnaught*np.exp(-alpha*t)+rbar*(1-np.exp(-alpha*t))+np.multiply(sigma,integral)

# rtN Euler-Maruyama
def rtNEM(N: int, B: list):
    rtN = [rnaught]
    i = 1
    while i < N:
        rtN.append((1-alpha*T/N)*rtN[i-1]+alpha*rbar*T/N+sigma*(B[i]-B[i-1]))
        i += 1
    return rtN

# rtN Milstein
def rtNM(N: int, B: list):
    rtN = [rnaught]
    i = 1
    while i < N:
        rtN.append(rtN[i-1]+alpha*(rbar-rtN[i-1])*T/N+sigma*(B[i]-B[i-1]))
        i += 1
    return rtN

# Absolute difference of rtNEM and rt
def absrtNEMminusrt(N: int, B: list):
    return np.abs(np.subtract(rtNEM(N,B),rt(N,B)))

# Max_t of absolute difference of rtNEM and rt
def maxAbsrtNEMminusrt(N: int, B: list):
    return max(np.abs(np.subtract(rtNEM(N,B),rt(N,B))))

# Absolute difference of rtNM and rt
def absrtNMminusrt(N: int, B: list):
    return np.abs(np.subtract(rtNM(N,B),rt(N,B)))

# Max_t of absolute difference of rtNM and rt
def maxAbsrtNMminusrt(N: int, B: list):
    return max(np.abs(np.subtract(rtNM(N,B),rt(N,B))))

# St exact simulation
def St(N: int, B: list): # St is geometric Brownian motion
    return snaught*np.exp((mu-sigma**2/2)*t+sigma*B)

# StN Euler-Maruyama
def StNEM(N: int, B: list):
    StN = [snaught]
    i = 1
    while i < N:
        StN.append(StN[i-1]*(1+mu*T/N+sigma*(B[i]-B[i-1])))
        i += 1
    return StN

# StN Milstein
def StNM(N: int, B: list):
    StN = [snaught]
    i = 1
    while i < N:
        StN.append(StN[i-1]+StN[i-1]*mu*T/N+sigma*StN[i-1]*(B[i]-B[i-1])+0.5*sigma**2*StN[i-1]*((B[i]-B[i-1])**2-T/N))
        i += 1
    return StN

# Absolute difference of StNEM and St
def absStNEMminusSt(N: int, B: list):
    return np.abs(np.subtract(StNEM(N,B),St(N,B)))

# Max_t of absolute difference of StNEM and St
def maxAbsStNEMminusSt(N: int, B: list):
    return max(np.abs(np.subtract(StNEM(N,B),St(N,B))))

# Absolute difference of StNM and St
def absStNMminusSt(N: int, B: list):
    return np.abs(np.subtract(StNM(N,B),St(N,B)))

# Max_t of absolute difference of StNM and St
def maxAbsStNMminusSt(N: int, B: list):
    return max(np.abs(np.subtract(StNM(N,B),St(N,B))))

# Run simulations from functions defined above
sims = 10000
Ert = []
ErtNEM = []
ErtNM = []
MaxAbsErtNEM = []
MaxEAbsrtNEM = []
EMaxAbsrtNEM = []
MaxAbsErtNM = []
MaxEAbsrtNM = []
EMaxAbsrtNM = []

ESt = []
EStNEM = []
EStNM = []
MaxAbsEStNEM = []
MaxEAbsStNEM = []
EMaxAbsStNEM = []
MaxAbsEStNM = []
MaxEAbsStNM = []
EMaxAbsStNM = []
i = 0
while i < len(Ns):
    j = 0
    B = []
    while j < sims:
        B.append(Bt(Ns[i]))
        j += 1
    t = np.linspace(0,1,Ns[i])

    Ert.append(expectation(sims,rt,Ns[i],B))
    ESt.append(expectation(sims,St,Ns[i],B))
    ErtNEM.append(expectation(sims,rtNEM,Ns[i],B))
    EStNEM.append(expectation(sims,StNEM,Ns[i],B))
    MaxAbsErtNEM.append(max(np.abs(np.subtract(Ert[i],ErtNEM[i]))))
    MaxAbsEStNEM.append(max(np.abs(np.subtract(ESt[i],EStNEM[i]))))
    ErtNM.append(expectation(sims,rtNM,Ns[i],B))
    EStNM.append(expectation(sims,StNM,Ns[i],B))
    MaxAbsErtNM.append(max(np.abs(np.subtract(Ert[i],ErtNM[i]))))
    MaxAbsEStNM.append(max(np.abs(np.subtract(ESt[i],EStNM[i]))))

    MaxEAbsrtNEM.append(max(expectation(sims,absrtNEMminusrt,Ns[i],B)))
    MaxEAbsStNEM.append(max(expectation(sims,absStNEMminusSt,Ns[i],B)))
    MaxEAbsrtNM.append(max(expectation(sims,absrtNMminusrt,Ns[i],B)))
    MaxEAbsStNM.append(max(expectation(sims,absStNMminusSt,Ns[i],B)))

    EMaxAbsrtNEM.append(expectation(sims,maxAbsrtNEMminusrt,Ns[i],B))
    EMaxAbsStNEM.append(expectation(sims,maxAbsStNEMminusSt,Ns[i],B))
    EMaxAbsrtNM.append(expectation(sims,maxAbsrtNMminusrt,Ns[i],B))
    EMaxAbsStNM.append(expectation(sims,maxAbsStNMminusSt,Ns[i],B))
    i += 1

# Create plots
plt.plot(Ns,MaxAbsEStNEM,label="MaxAbsEStNEM")
plt.plot(Ns,MaxAbsErtNEM,label="MaxAbsErtNEM")
plt.plot(Ns,MaxAbsEStNM,label="MaxAbsEStNM")
plt.plot(Ns,MaxAbsErtNM,label="MaxAbsErtNM")
plt.plot(Ns,np.divide(1,Ns),ls='--',label="1/N")
plt.plot(Ns,np.divide(1,np.sqrt(Ns)),ls='--',label="1/sqrt(N)")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("weak convergence")
plt.show()



plt.plot(Ns,MaxEAbsStNEM,label="MaxEAbsStNEM")
plt.plot(Ns,MaxEAbsrtNEM,label="MaxEAbsrtNEM")
plt.plot(Ns,MaxEAbsStNM,label="MaxEAbsStNM")
plt.plot(Ns,MaxEAbsrtNM,label="MaxEAbsrtNM")
plt.plot(Ns,np.divide(1,Ns),ls='--',label="1/N")
plt.plot(Ns,np.divide(1,np.sqrt(Ns)),ls='--',label="1/sqrt(N)")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("strong convergence")
plt.show()



plt.plot(Ns,EMaxAbsStNEM,label="EMaxAbsStNEM")
plt.plot(Ns,EMaxAbsrtNEM,label="EMaxAbsrtNEM")
plt.plot(Ns,EMaxAbsStNM,label="EMaxAbsStNM")
plt.plot(Ns,EMaxAbsrtNM,label="EMaxAbsrtNM")
plt.plot(Ns,np.divide(1,Ns),ls='--',label="1/N")
plt.plot(Ns,np.divide(1,np.sqrt(Ns)),ls='--',label="1/sqrt(N)")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("pathwise convergence")
plt.show()