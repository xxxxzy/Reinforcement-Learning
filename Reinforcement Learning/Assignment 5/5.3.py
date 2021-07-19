import random, math
import matplotlib.pyplot as plt
import numpy as np



"""
UCB
"""

mu1 = 0.5
mu2 = 0.25

T = 10000
N = 10 #experience 10 times

def sample_from_distribution(mu):
    
    val = random.random()
    
    if val <= mu:
        return 1
    else:
        return 0


ucb_regret = np.zeros((T, N))
ucb_num_op_arms = np.zeros((T, N))
iterations = [0] * T
best_mu = max(mu1, mu2)

num=0
for it in range(N):
    mu1_cur = 0
    mu2_cur = 0
    t1_cur = 0
    t2_cur = 0
    total_reward = 0
    total_best_reward = 0

    mu1_cur = sample_from_distribution(mu1)
    mu2_cur = sample_from_distribution(mu2)
    t1_cur = 1
    t2_cur = 1

    for t in range(T):
        
        mu1_ucb = mu1_cur + math.sqrt(3 * math.log(t+1) / (2 * t1_cur))
        mu2_ucb = mu2_cur + math.sqrt(3 * math.log(t+1) / (2 * t2_cur))
        
        if mu1_ucb > mu2_ucb:
            
            new_sample = sample_from_distribution(mu1)
            mu1_cur = (t1_cur * mu1_cur + new_sample) * 1.0 / (t1_cur + 1)
            
            if t == 0:
                ucb_num_op_arms[t][it] = 1
                
            else:
                ucb_num_op_arms[t][it] = ucb_num_op_arms[t-1][it] + 1
            t1_cur += 1
            total_best_reward += new_sample

        else:
            
            new_sample = sample_from_distribution(mu2)
            mu2_cur = (t2_cur * mu2_cur + new_sample) * 1.0 / (t2_cur + 1)
            
            if t == 0:
                ucb_num_op_arms[t][it] = 0
                
            else:
                ucb_num_op_arms[t][it] = ucb_num_op_arms[t-1][it]
            t2_cur += 1
            total_best_reward += sample_from_distribution(mu1)
            num = num+1

        total_reward += new_sample
        ucb_regret[t][it] = total_best_reward - (total_reward * 1.0)
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
#plt.show()


"""
EXP3
"""

cum_loss = np.zeros(16)
exp3_regret = np.zeros((T, N))
exp3_num_op_arms = np.zeros((T, N))
eta = 0.0085
total_time = 10000


for it in range(N):
    total_reward = 0
    total_best_reward = 0
    cum_loss.fill(0.0)
#    print(cum_loss)

    for t in range(T):
        
        eta = np.sqrt(np.log(2)/(2*t))


        p = np.exp(eta * cum_loss) / np.sum(np.exp(eta * cum_loss))
        print(p)
        arm_num = sample_from_distribution(p[0])

        if arm_num == 1:
            new_sample = sample_from_distribution(mu1)
            cum_loss[0] += (new_sample) / p[0]

            if t == 0:
                exp3_num_op_arms[t][it] = 1
            else:
                exp3_num_op_arms[t][it] = exp3_num_op_arms[t-1][it] + 1
            total_best_reward += new_sample
        else:
            new_sample = sample_from_distribution(mu2)
            cum_loss[1] += new_sample / p[1]
            if t == 0:
                exp3_num_op_arms[t][it] = 0
            else:
                exp3_num_op_arms[t][it] = exp3_num_op_arms[t-1][it]
            total_best_reward += sample_from_distribution(mu1)

        total_reward += new_sample
        exp3_regret[t][it] = total_best_reward - total_reward
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

