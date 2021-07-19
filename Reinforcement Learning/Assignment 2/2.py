import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
import math
import time
import matplotlib.pylab as plt 
import matplotlib.pyplot as mp
from pylab import*
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes



"""
Load data
"""

def label(s):
    it={b'g':0, b'b':1 }  #change into numbers
    return it[s]

data=np.loadtxt('ionosphere.data', dtype=float, delimiter=',', converters={34:label} )
#print(type(data))

x,y=np.split(data,indices_or_sections=(34,),axis=1)  #x:data y:label
train_data,test_data,train_label,test_label = train_test_split(x,y, random_state=0, train_size=200/351,test_size=151/351)
# Now, the train set:200, test set:151
#def PAC_train(train_data,train_label):
    
r_data,val_data,r_label,val_label = train_test_split(train_data,train_label, random_state=0, train_size=35/200,test_size=165/200)
    
#    return r_data,val_data,r_label,val_label

#print(type(PAC_train(train_data,train_label)))

"""
Parameter of SVM:
    C: log_{10}C \in {-3,-2,...,3}
    Gamma:
        G(X_i)=min_{(X_i,Y_i)\in S\capY_i\Y_j}|X_i-X_j|
        gamma_J=1/(2*medianG^2)
        Gamma \in {10^{-4}gamma_J,10^{-2}gamma_J,gamma_J,10^{2}gamma_J,10^{4}gamma_J}
"""


#calculate G(X)
label_0=[]
label_1=[]

for i in range (len(train_label)):
    
    if train_label[i] == 0:
        
        label_0.append(i)
        
    else:
        
        label_1.append(i)
        
G=np.zeros(len(label_0))
       
for i in range(len(label_0)):
    
    V=100
    
    for j in range(len(label_1)):
        
        X = np.sum(np.abs(train_data[label_0[i]]-train_data[label_1[j]]))
        V = min(V,X)
        
    G[i]=V

med = np.median(G)
gamma_J = 1/(med*med*2)

Gamma = [gamma_J*0.0001,gamma_J*0.01,gamma_J,gamma_J*100,gamma_J**10000]
C = [0.001,0.01,0.1,1,10,100,1000] 

param_grid = {"gamma":[gamma_J*0.0001,gamma_J*0.01,gamma_J,gamma_J*100,gamma_J**10000],"C":[0.001,0.01,0.1,1,10,100,1000]}


"""
CV SVM
"""


start =time.clock() #run time start

grid_search = GridSearchCV(SVC(kernel='rbf'),param_grid,cv=5)
grid_search.fit(train_data,train_label)  # get the parameters
#print("Best parameters:{}".format(grid_search.best_params_))


h1 = SVC(C=10, kernel='rbf', gamma=gamma_J*100)
h1.fit(train_data, train_label)
    
pre_label = h1.predict(test_data)

#print("Accuracy：", accuracy_score(test_label,pre_label))
#c = 1-accuracy_score(test_label,pre_label)

end = time.clock() #time end
print('SVM time: %s Seconds'%(end-start))



"""
Method
"""

rho_down = 0
val_minloss = 100
eps = 1e-10
p_2 = []
rho_up = []
e = []

start =time.clock()
for m in range(1,101):
 
#    start =time.clock()
    r_data,val_data,r_label,val_label = train_test_split(train_data,train_label, train_size=35/200,test_size=165/200)

    n_val = 200 - 35
    delta = 0.05
    lamda = np.sqrt(np.log(2*n_val/delta)/n_val)
    pi = 1/m
        
    h2 = svm.SVC(kernel='rbf', gamma=gamma_J*100)
    scores = 1-cross_val_score(h2, val_data, val_label, cv=2)
    val_loss = np.sum(scores)/2
    val_minloss = min(val_loss,val_minloss)

    
    rho_up = np.append(rho_up,np.exp(-lamda*n_val*(val_loss-val_minloss)))       
    rho_down = np.sum(rho_up)
    rho = rho_up[m-1]/rho_down

    end = time.clock()
    
    print('SVM time: %s Seconds'%(end-start))
        
    e = np.append(e,val_loss/(1-lamda/2)+((rho-pi)**2*0.5+np.log(2*np.sqrt(n_val)/delta))/(lamda*n_val*(1-lamda/2)))
#    print(e)

    
   
    bound = ((rho-pi)**2*0.5+np.log(2*np.sqrt(n_val)/delta))/n_val
    
    y = (1+val_loss)/2
    
    step = (1-val_loss)/4
#    print(step)

    if val_loss>0:
        p = val_loss
    else:
        p = 1

    while step > eps:
    
        if val_loss*math.log(p/y) + (1-val_loss)*math.log((1-val_loss)/(1-y)) < bound:
            
            y = y+step

        else:
            y = y-step

    
        step = step/2
    
    if y < 1:
    
        p_2 = np.append(p_2,y)
        
#print(p_2)
           
'''
plt.xlabel("m")
plt.ylabel("loss")   
plt.plot([1,5,6,9,14,19,24,27,30,40,46,49,50,56,60,66,70,80,90,99], p_2)    
plt.grid(linestyle='--')
#plt.legend(labels = ['Hoeffding','kl-inequality','Pinsker relaxation','Refined Pinsker'],loc = 'upper left')
plot = plt.show()

'''
x=[1,5,6,9,14,19,24,27,30,40,46,49,50,56,60,66,70,80,90,99]
a=[]
b=[]
c=np.zeros(20)+0.0794
d=np.zeros(20)+0.72
print(c)
for i in [1,5,6,9,14,19,24,27,30,40,46,49,50,56,60,66,70,80,90,99]:
    a = np.append(a,p_2[i])
    b = np.append(b,e[i])



plt.xlabel("m")
plt.ylabel("Test loss/Run time") 
f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.yaxis.tick_left()
plt.plot(x,a)
plt.plot(x,b)
plt.plot(x,c)
plt.plot(x,d,linestyle='--')
plt.legend(labels = ['Our Method','Bound','SVM','t_cv'],loc = 'upper right')
plt.xlabel("m")
plt.ylabel("Test loss/Run time") 
plt.show()


















