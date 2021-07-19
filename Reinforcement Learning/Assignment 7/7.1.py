import gym
import numpy as np
import matplotlib.pyplot as plt
import cma
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Define task
env = gym.make('CartPole-v0')
state_space_dimension = 4
action_space_dimension = 1
#4*10+10*2=60
def nn(Weights_L1,Weights_L2):
    
    state = tf.placeholder(tf.float32, shape=[None, 4])
    action = tf.placeholder(tf.int32, shape=[None])


    Wx_plus_B_L1 = tf.matmul(state, Weights_L1)
    L1 = tf.nn.tanh(Wx_plus_B_L1)   


    Wx_plus_B_L2 = tf.matmul(L1, Weights_L2)
    prediction = tf.nn.tanh(Wx_plus_B_L2)  
       #预测值
    return L1,prediction 

        
def fitness_cart_pole(x, nn, env, k=4):
    '''
    Returns negative accumulated reward for single pole, fully environment.

    Parameters:
        x: Parameter vector encoding the weights.
        nn: Parameterized model.
        env: Environment ('CartPole-v0').
        k: Number of addional balancing trials after the first succeeded.
    '''
    # Do a single balancing experiment
    Weights_L1 = tf.Variable(tf.random_normal([1, 10]))    #随机产生10个权重
    Weights_L2 = tf.Variable(tf.random_normal([10, 1]))  
    nn(Weights_L1,Weights_L2)
    
    def fitness_cart_pole_run(nn, env):
        state = env.reset()  # Forget about previous episode
#        print(nn(state.reshape((1, state_space_dimension))))
        R = 0  # Accumulated reward
        while True:
            out = nn.L1
            a = int(out > 0)
            state, reward, done, _ = env.step(a)  # Simulate pole
            R += reward  # Accumulate 
            if done:
                return R  # Episode ended
 
    weights = nn.prediction  # Bring parameter vector into weight structure
#    nn.set_weights(weights)  # Set the policy parameters
    
    R = fitness_cart_pole_run(nn, env)
    
    if(R == 200):  # Managed to balance the pole once
        for i in range(k):  # Try addional k times
            R_prime = fitness_cart_pole_run(nn, env)
            if(R_prime<200):  # One of the k additional trials failed
                return -R  # We consider minimization 
        return -1000  # Success, managed to balance k+1 times, high reward (negative because we consider minimization)
    return -R  # We consider minimization    

# Generate initial search point and initial hidden RNN states
initial_weights = np.random.normal(0, 0.01, 50)  # Random parameters for initial policy, d denotes the number of weights
initial_sigma = .01  # Initial global step-size sigma

#def policy_net():
    
# Do the optimization
res = cma.fmin(fitness_cart_pole,  # Objective function
               initial_weights,  # Initial search point
               initial_sigma,  # Initial global step-size sigma
               args=([nn, env]),  # Arguments passed to the fitness function
               options={'ftarget': -999.9})

# Learn even more on CMA-ES
cma.CMAOptions() 
#cma.fmin?