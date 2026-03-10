import numpy as np
import matplotlib.pyplot as plt

alpha = 3
sigma = 0.25
rbar = 0.05
rnaught = 0.05
snaught = 100
mu = 0.15
Ns = [1,5,10,50,100,500,1000]
N = max(Ns)
T = 1

t = np.linspace(0,T,N)

def expectation(sims: int, expectee, N):
    trials = []
    i = 0
    while i < sims:
        trials.append(expectee(N,Bt(N)))
        i += 1
    
    return np.average(trials,axis=0)

def Bt(N: int):
    Z = np.random.normal(0,1,N)
    dt = np.ones(N) * T/N
    return np.cumsum(Z*np.sqrt(dt))

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

def rtNEM(N: int, B: list):
    rtN = [rnaught]
    i = 1
    while i < N:
        rtN.append((1-alpha*T/N)*rtN[i-1]+alpha*rbar*T/N+sigma*(B[i]-B[i-1]))
        i += 1
    return rtN

def rtNM(N: int, B: list):
    rtN = [rnaught]
    i = 1
    while i < N:
        rtN.append(rtN[i-1]+alpha*(rbar-rtN[i-1])*T/N+sigma*(B[i]-B[i-1]))
        i += 1
    return rtN

def absrtNEMminusrt(N: int, B: list):
    return np.abs(np.subtract(rtNEM(N,B),rt(N,B)))

def maxAbsrtNEMminusrt(N: int, B: list):
    return max(np.abs(np.subtract(rtNEM(N,B),rt(N,B))))

def absrtNMminusrt(N: int, B: list):
    return np.abs(np.subtract(rtNM(N,B),rt(N,B)))

def maxAbsrtNMminusrt(N: int, B: list):
    return max(np.abs(np.subtract(rtNM(N,B),rt(N,B))))

def St(N: int, B: list): # St is geometric Brownian motion
    return snaught*np.exp((mu-sigma**2/2)*t+sigma*B)

def StNEM(N: int, B: list):
    StN = [snaught]
    i = 1
    while i < N:
        StN.append(StN[i-1]*(1+mu*T/N+sigma*(B[i]-B[i-1])))
        i += 1
    return StN

def StNM(N: int, B: list):
    StN = [snaught]
    i = 1
    while i < N:
        StN.append(StN[i-1]+StN[i-1]*mu*T/N+sigma*(B[i]-B[i-1]))
        i += 1
    return StN

def absStNEMminusSt(N: int, B: list):
    return np.abs(np.subtract(StNEM(N,B),St(N,B)))

def maxAbsStNEMminusSt(N: int, B: list):
    return max(np.abs(np.subtract(StNEM(N,B),St(N,B))))

def absStNMminusSt(N: int, B: list):
    return np.abs(np.subtract(StNM(N,B),St(N,B)))

def maxAbsStNMminusSt(N: int, B: list):
    return max(np.abs(np.subtract(StNM(N,B),St(N,B))))

B = Bt(N)

sims = 100
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
    t = np.linspace(0,1,Ns[i])
    Ert.append(expectation(sims,rt,Ns[i]))
    ESt.append(expectation(sims,St,Ns[i]))
    ErtNEM.append(expectation(sims,rtNEM,Ns[i]))
    EStNEM.append(expectation(sims,StNEM,Ns[i]))
    MaxAbsErtNEM.append(max(np.abs(np.subtract(Ert[i],ErtNEM[i]))))
    MaxAbsEStNEM.append(max(np.abs(np.subtract(ESt[i],EStNEM[i]))))
    ErtNM.append(expectation(sims,rtNM,Ns[i]))
    EStNM.append(expectation(sims,StNM,Ns[i]))
    MaxAbsErtNM.append(max(np.abs(np.subtract(Ert[i],ErtNM[i]))))
    MaxAbsEStNM.append(max(np.abs(np.subtract(ESt[i],EStNM[i]))))

    MaxEAbsrtNEM.append(max(expectation(sims,absrtNEMminusrt,Ns[i])))
    MaxEAbsStNEM.append(max(expectation(sims,absStNEMminusSt,Ns[i])))
    MaxEAbsrtNM.append(max(expectation(sims,absrtNMminusrt,Ns[i])))
    MaxEAbsStNM.append(max(expectation(sims,absStNMminusSt,Ns[i])))

    EMaxAbsrtNEM.append(expectation(sims,maxAbsrtNEMminusrt,Ns[i]))
    EMaxAbsStNEM.append(expectation(sims,maxAbsStNEMminusSt,Ns[i]))
    EMaxAbsrtNM.append(expectation(sims,maxAbsrtNMminusrt,Ns[i]))
    EMaxAbsStNM.append(expectation(sims,maxAbsStNMminusSt,Ns[i]))
    i += 1

plt.plot(Ns,MaxAbsEStNEM,label="MaxAbsEStNEM")
plt.plot(Ns,MaxEAbsStNEM,label="MaxEAbsStNEM")
plt.plot(Ns,EMaxAbsStNEM,label="EMaxAbsStNEM")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("St Euler-Maruyama")
plt.show()

plt.plot(Ns,MaxAbsErtNEM,label="MaxAbsErtNEM")
plt.plot(Ns,MaxEAbsrtNEM,label="MaxEAbsrtNEM")
plt.plot(Ns,EMaxAbsrtNEM,label="EMaxAbsrtNEM")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("rt Euler-Maruyama")
plt.show()

plt.plot(Ns,MaxAbsEStNM,label="MaxAbsEStNM")
plt.plot(Ns,MaxEAbsStNM,label="MaxEAbsStNM")
plt.plot(Ns,EMaxAbsStNM,label="EMaxAbsStNM")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("St Milstein")
plt.show()

plt.plot(Ns,MaxAbsErtNM,label="MaxAbsErtNM")
plt.plot(Ns,MaxEAbsrtNM,label="MaxEAbsrtNM")
plt.plot(Ns,EMaxAbsrtNM,label="EMaxAbsrtNM")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title("rt Milstein")
plt.show()