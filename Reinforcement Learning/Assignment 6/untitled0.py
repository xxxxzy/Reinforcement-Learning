import math
import matplotlib.pyplot as plt
import numpy as np

"""
UCB
"""
data=np.loadtxt("data_preprocessed_features.txt")
data = data[data[:,0].argsort()]
action = data[:,0].tolist()
reward = data[:,1].tolist()

K=2
emp_prob = []

for i in range (0,K):

    num_k = action.count(i)
    reward_k = reward[0:num_k].count(1) #each actions' rewards
    del reward[0:num_k]
    emp_prob = np.append(emp_prob,reward_k / num_k)
    
#emp_prob=emp_prob.tolist()
emp_prob = [0.04984787134262119,0.01191586128482092]
mu = dict(zip(list(range(1, K+1)), emp_prob))



T = 10000
N = 10 #experience 10 times



ucb_regret = np.zeros((T, N))
ucb_num_op_arms = np.zeros((T, N))
iterations = [0] * T
mu1 = sorted(mu.items(), key=lambda d: d[1], reverse=True)
print(mu1)

best_mu =  mu1[0][1]

num=0

for it in range(N):
    count = dict(zip(list(range(1, K+1)), [1] * K)) 
    reward = dict(zip(list(range(1, K+1)), [0] * K)) 
    mu_hat = mu

    total_reward = 0
    total_best_reward = 0

    mu_ucb = dict(zip(list(range(1, K+1)), [0] * K))
    
    for t in range(T):

        for i in range(1,K+1):

            mu_ucb[i] = mu_hat[i] + math.sqrt(3 * math.log(t+1) / (2 * count[i]))

        mu_ucb1 = sorted(mu_ucb.items(), key=lambda d: d[1], reverse=True)
        best_arm = mu_ucb1[0][0]

        mu = dict(zip(list(range(1, K+1)), emp_prob))
        reward[best_arm] = np.random.choice([1, 0], p=[mu[best_arm], 1 - mu[best_arm]])

        mu_hat[best_arm] = (count[best_arm]*mu_hat[best_arm]+best_arm*reward[best_arm])/(count[best_arm]+1)
        count[best_arm] = count[best_arm]+1
        


        total_reward += reward[best_arm]
        total_best_reward = count[best_arm]*mu[best_arm]
        ucb_regret[t][it] = - total_best_reward + total_reward
        iterations[t] = t


avg_ucb_regret = [0] * T
std_ucb_regret = [0] * T

for i in range(T):

    avg_ucb_regret[i] = np.average(ucb_regret[i])
    std_ucb_regret[i] = np.std(ucb_regret[i])


ucb_avg_op_arms = [0] * T
std_ucb_op_arms = [0] * T
for i in range(T):
    ucb_avg_op_arms[i] = np.average( ucb_num_op_arms[i] ) / (i+1)
    std_ucb_op_arms[i] = np.std(ucb_avg_op_arms[i] / (i+1))


avg=np.array(ucb_avg_op_arms)
std = np.array(std_ucb_op_arms)


plt.plot(iterations, avg_ucb_regret, label='UCB_Ave')
plt.plot(iterations, avg+std, label='UCB_Ave+Std')
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.legend()
plt.show()



"""
EXP3
"""

cum_loss = np.zeros(K)
exp3_regret = np.zeros((T, K))
exp3_num_op_arms = np.zeros((T, K))

for it in range(N):
    total_reward = 0
    total_best_reward = 0
    cum_loss.fill(0.0)

    count = dict(zip(list(range(1, K+1)), [1] * K)) 
    reward = dict(zip(list(range(1, K+1)), [0] * K)) 

    for t in range(T):

        eta =0.0085

        mu = dict(zip(list(range(1, K+1)), emp_prob))

        p = np.exp(eta * cum_loss) / np.sum(np.exp(eta * cum_loss))
       
        arm_num = np.random.choice([1, 0], p=[1 - p[0], p[0]])
        
        for i in range(1,K+1):
            
            mu = dict(zip(list(range(1, K+1)), emp_prob))
            if arm_num == 1:
            
                reward[i] = np.random.choice([1, 0], p=[1 - mu[i], mu[i]])

                cum_loss[0] += (reward[i]*i) / p[0]

                if t == 0:
                    exp3_num_op_arms[t][it] = 1
                else:
                    exp3_num_op_arms[t][it] = exp3_num_op_arms[t-1][it] + 1
                total_best_reward += reward[i]

            total_reward += reward[i]
        exp3_regret[t][it] = - total_best_reward + total_reward
        iterations[t] = t



avg_exp3_regret = [0] * T
std_regret = [0] * T
for i in range(T):

    avg_exp3_regret[i] = np.average(exp3_regret[i])
    std_regret[i] = np.std(exp3_regret[i]/10)
    if i%100 != 0:
        std_regret[i] = 0


exp3_avg_op_arms = [0] * T
std_op_arms = [0] * T
for i in range(T):
    exp3_avg_op_arms[i] = np.average( exp3_num_op_arms[i] ) / (i+1)
    std_op_arms[i] = np.std(exp3_num_op_arms[i])/(10*(i+1))
    if i%100 != 0:
        std_op_arms[i] = 0

avg_exp3=np.array(avg_exp3_regret)
std_exp3 = np.array(std_regret)


plt.plot(iterations, avg_exp3_regret, label='EXP3_Ave')
plt.plot(iterations, avg_exp3+std_exp3,label='EXP3_Ave+Std')
plt.xlabel('Iterations')

plt.ylabel('Regret')
plt.legend()
plt.show()

