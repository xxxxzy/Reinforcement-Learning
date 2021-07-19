
import numpy as np
import matplotlib.pyplot as plt


def ftl(mu):
    delta = 1-2*mu
    num = 0
    rt = []

    for t in range (1,1001):
        
        num = delta*1000*np.exp(-delta**2*t/2)
        
        rt = np.append(rt,num)
        
    return rt
#ftl(0.25)

def hedge_regret(eta,mu,T):
    
    re = []

    
    for t in range (1,1001):
        
        w_1 = np.exp(-eta*mu*t)
        w_2 = np.exp(-eta*t*(1-mu))
        
        r = (w_1/(w_1+w_2))*mu*T+(w_2/(w_1+w_2))*(1-mu)*T - T*mu
        re = np.append(re,r)
#    print(len(re))
    return re

def Anytime_hedge(mu,T,par):
    
    rp = []

    
    for t in range (1,1001):
        
        eta = par*np.sqrt(np.log(2)/t)
        
        w_1 = np.exp(-eta*mu*t)
        w_2 = np.exp(-eta*t*(1-mu))
        
        r = (w_1/(w_1+w_2))*mu*T+(w_2/(w_1+w_2))*(1-mu)*T - T*mu
        
        rp = np.append(rp,r)
        
    return rp
    
def average(mu,eta,T,par):
    
    ftlave = np.zeros(1000)
    hedave = np.zeros(1000)
    anyave =  np.zeros(1000)
    
    for i in range (0,10):
        
        ftlave += ftl(mu)
        hedave += hedge_regret(eta,mu,T)
        anyave += Anytime_hedge(mu,T,par)
        
    ftlave = ftlave/10
    hedave = hedave/10
    anyave = anyave/10
    return ftlave,hedave,anyave

#r = regret(np.sqrt(2*np.log(2)/1000),0.75,1000)
x = np.linspace(1,1000,num=1000)


plt.plot(x,ftl(0.25))
plt.plot(x,hedge_regret(np.sqrt(2*np.log(2)/1000),0.25,1000))
plt.plot(x,hedge_regret(np.sqrt(8*np.log(2)/1000),0.25,1000))
plt.plot(x,Anytime_hedge(0.25,1000,1))
plt.plot(x,Anytime_hedge(0.25,1000,2))

plt.xlabel('t')
plt.ylabel('pseudo regret')
plt.grid(linestyle='--')
plt.legend(labels=['FTL','Hedge with eta=sqrt(2lnK/T)','Hedge with eta=sqrt(8lnK/T)','Anytime Hedge with eta=sqrt(lnK/T)','Anytime Hedge with eta=2sqrt(lnK/T)']) 
plt.show()

plt.plot(x,ftl(0.375))
plt.plot(x,hedge_regret(np.sqrt(2*np.log(2)/1000),0.375,1000))
plt.plot(x,hedge_regret(np.sqrt(8*np.log(2)/1000),0.375,1000))
plt.plot(x,Anytime_hedge(0.375,1000,1))
plt.plot(x,Anytime_hedge(0.375,1000,2))

plt.xlabel('t')
plt.ylabel('pseudo regret')
plt.grid(linestyle='--')
plt.legend(labels=['FTL','Hedge with eta=sqrt(2lnK/T)','Hedge with eta=sqrt(8lnK/T)','Anytime Hedge with eta=sqrt(lnK/T)','Anytime Hedge with eta=2sqrt(lnK/T)']) 
plt.show()

plt.plot(x,ftl(0.4375))
plt.plot(x,hedge_regret(np.sqrt(2*np.log(2)/1000),0.4375,1000))
plt.plot(x,hedge_regret(np.sqrt(8*np.log(2)/1000),0.4375,1000))
plt.plot(x,Anytime_hedge(0.4375,1000,1))
plt.plot(x,Anytime_hedge(0.4375,1000,2))

plt.xlabel('t')
plt.ylabel('pseudo regret')
plt.grid(linestyle='--')
plt.legend(labels=['FTL','Hedge with eta=sqrt(2lnK/T)','Hedge with eta=sqrt(8lnK/T)','Anytime Hedge with eta=sqrt(lnK/T)','Anytime Hedge with eta=2sqrt(lnK/T)']) 
plt.show()