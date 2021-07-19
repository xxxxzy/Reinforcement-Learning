import math
import matplotlib.pyplot as plt
import numpy as np



def LowerBound(delta,n,p_hat):
    
    eps = 0.00001

    p_1 = []
    p_2 = []
    p_3 = []

    bound_1 = math.sqrt(math.log(1/delta)/(2*n))
    bound_2 = math.log((n+1)/delta)/n
    
    
 
    """
    Hoeffding's lower bound
    """

    for i in range (len(p_hat)):

        if p_hat[i]<0 and p_hat[i]>1:
    
            print("wrong argument")
            break
    
        if p_hat[i]-bound_1 >= 1: 
    
            break

        if p_hat[i]-bound_1 >= 0:
    
            p_1 = np.append(p_1,p_hat[i]-bound_1)  
        
     
    
    """
    kl-inequality lower bound
    """

    for i in range (len(p_hat)):

        if p_hat[i]<0 and p_hat[i]>1 and bound_2<0:
            
            print("wrong argument")
            break

        if bound_2 == 0:
            
            y = p_hat[i]

        y = (1+p_hat[i])/2
        step = (1-p_hat[i])/4

        if p_hat[i]>0:
            
            p = p_hat[i]
            
        else:
            
            p = 1

        while step > eps:
    
            if p_hat[i]*math.log(p/y) + (1-p_hat[i])*math.log((1-p_hat[i])/(1-y)) < bound_2:
                #we still should kl(p^//p) smaller than z, but this time we will minimize the y

                y = y-step 
                #Differnt with upper bound
                #If kl(p^//p)< z, p-step to become smaller
            
            else:
                
                y = y+step
                #If not, p+step to make kl(p^//p) satisfy the condition, smaller than z

            step = step/2
    
        if y < 1:
    
            p_2 = np.append(p_2,y)



    
    """
    kl-inequality lower bound
    """

    for i in range (len(p_hat)):

        if p_hat[i]<0 and p_hat[i]>1 and bound_2<0:
            
            print("wrong argument")
            break

        if bound_2 == 0:
            
            y = p_hat[i]

        y = (1+p_hat[i])/2
        step = (1-p_hat[i])/4

        if p_hat[i]>0:
            
            p = p_hat[i]
            
        else:
            
            p = 1

        while step > eps:
    
            if p_hat[i]*math.log(p/y) + (1-p_hat[i])*math.log((1-p_hat[i])/(1-y)) < bound_2:
                #we still should kl(p^//p) smaller than z, but this time we will minimize the y

                y = y+step 
                #Differnt with upper bound
                #If kl(p^//p)< z, p-step to become smaller
            
            else:
                
                y = y-step
                #If not, p+step to make kl(p^//p) satisfy the condition, smaller than z

            step = step/2
    
        if y < 1:
    
            p_3 = np.append(p_3,y)
            
            

    plt.xlabel("p^")
    plt.ylabel("p")
    plt.grid(linestyle='--') 
    plt.plot(p_hat[0:len(p_1)], p_1)      
    plt.plot(p_hat[0:len(p_2)], p_2)
    plt.plot(p_hat[0:len(p_3)], p_3)
    plt.legend(labels = ['Hoeffding','kl-inequality lower','kl-inequality upper'], loc = 'lower right')
    plot = plt.show()
    
    return plot

delta = 0.1
n = 1000
p_hat = list(np.arange(0, 1, 0.001))
p_hatbig = list(np.arange(0.9, 1, 0.001))

LowerBound(delta,n,p_hat)
LowerBound(delta,n,p_hatbig)



