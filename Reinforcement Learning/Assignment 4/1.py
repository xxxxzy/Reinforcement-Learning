import gym
import gym_gridworlds # pip install gym-gridworlds
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt



env_name = 'CliffWalking-v0'
env = gym.make(env_name)
env.render()
number_of_actions = env.action_space.n


number_of_states = env.observation_space.n
print("|S| =", number_of_states)
print("|A| =", number_of_actions)


a = env.action_space.sample()


"""
updates the action-value function estimate using the most recent time step
"""
def update_Q(Qsa,Qsa_next,reward,alpha,gamma):
   
    return Qsa + alpha*(reward + (gamma * Qsa_next) - Qsa)


"""
obtains the action probabilities corresponding to epsilon-greedy policy
""" 
def epsilion_greedy_probs(env,Q_s,i_episode,eps=None):

    epsilon = 0.1
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(number_of_actions) * epsilon/number_of_actions
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon/number_of_actions)
    return policy_s


"""
q-learning algorithm
"""
def q_learning(env, num_episodes, alpha, gamma=1.0):
   
    Q = defaultdict(lambda: np.zeros(number_of_actions))  #initialize empty dictionary of arrays
 
    #initialize performance monitor
    plot_every = 1
    tmp_scores = deque(maxlen=plot_every)
    tmp_step = []
    scores = deque(maxlen=num_episodes)
    step = deque(maxlen=num_episodes)
    
    for i_episode in range(1, num_episodes+1): #loop over episodes
        
        
        score = 0 #initialize score
        step = 0
        
        # begin an episode, observe S
        s = env.reset()
        while True:
           
            policy_s = epsilion_greedy_probs(env,Q[s],i_episode)  #get epsilon-greedy action probabilities
            action = np.random.choice(np.arange(number_of_actions),p=policy_s) #pick next action A
            #action = env.action_space.sample()
            next_state,reward,done,info = env.step(action) #take action A, observe R, S'
            score += reward #add reward to score
            step += 1
            Q[s][action] = update_Q(Q[s][action],np.max(Q[next_state]),\
                                                               reward,alpha,gamma) #update Q
            s = next_state

            if done:

                tmp_scores.append(score)
                tmp_step.append(step)
                
                break
            
        if (i_episode % plot_every == 0):
            
            scores.append(np.mean(tmp_scores))

    
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Accumulated Reward')
    plt.grid(linestyle='--') 
    plt.show()   


    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),tmp_step)
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Steps')
    plt.grid(linestyle='--') 
    plt.show()

    return Q
   
    
"""
sarsa algorithm
"""
def sarsa(env, num_episodes, alpha, gamma=1.0):

    Q = defaultdict(lambda: np.zeros(number_of_actions))

    plot_every = 1
    tmp_scores = deque(maxlen=plot_every)
    tmp_step = []
    scores = deque(maxlen=num_episodes)
    step = deque(maxlen=num_episodes)
    
    for i_episode in range(1, num_episodes+1):

        score = 0
        step = 0
        state = env.reset()
        policy_s = epsilion_greedy_probs(env,Q[state],i_episode)
        action = np.random.choice(np.arange(number_of_actions),p=policy_s)
        
        while True:

            next_state,reward,done,info = env.step(action)

            score += reward
            step += 1
            
            if not done:

                policy_s = epsilion_greedy_probs(env,Q[next_state],i_episode)
                next_action = np.random.choice(np.arange(number_of_actions),p=policy_s)
                Q[state][action] = update_Q(Q[state][action],Q[next_state][next_action],
                                         reward,alpha,gamma)
                state = next_state
                action = next_action
                
            if done:
              
                Q[state][action] = update_Q(Q[state][action],0,reward,alpha,gamma)
                tmp_scores.append(score)
                tmp_step.append(step)
                
                break

        
        if(i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))

    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Accumulated Reward')
    plt.grid(linestyle='--') 
    plt.show()

  
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),tmp_step)
    plt.xlabel('Episode Number')
    plt.ylabel('Number of Steps')
    plt.grid(linestyle='--') 
    plt.show() 
   
#    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q

Q_learning = q_learning(env, 2000, .1)
Sarsa = sarsa(env, 2000, .1)

V_q = np.array([np.max(Q_learning[key]) if key in Q_learning else 0 for key in np.arange(48)])
print((np.around(V_q)).reshape((4,12)))

V_sarsa = np.array([np.max(Sarsa[key]) if key in Sarsa else 0 for key in np.arange(48)])
print((np.around(V_sarsa)).reshape((4,12)))









