import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')



class Memory():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self,experience):
        self.buffer.append(experience)
    
    def sample(self, batch):
        sample_size = min(len(self.buffer), batch)
        samples = random.choices(self.buffer, k=sample_size)
        
        return map(list, zip(*samples))
    
    
class QNetwork():
    def __init__(self, state_dim, action_size):
        
        #define state, action and hat(Q)
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        
        #transfer to one-hot type matrix
        action_one_hot = tf.one_hot(self.action_in, depth= action_size)
        
        self.hidden = tf.layers.dense(self.state_in, 64, activation=tf.nn.relu) #unit = 64
        self.q_state = tf.layers.dense(self.hidden, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
       
        
        #the error for a batch of size 100 
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        
        #train the model
        self.optimizer = tf.train.AdamOptimizer(learning_rate =0.001, epsilon=0.01).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        
        session.run(self.optimizer, feed_dict = feed)
    
    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict = {self.state_in: state})
        
        return q_state
    


class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = 2
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.99
        self.replay= Memory(maxlen=10000)

        #eps=eps_min+(eps_max-eps_min)*exp(-0.001*total step)
        self.eps = 1.0

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        #eps_greedy
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        
        action = action_random if random.random() < self.eps else action_greedy
        
        return action

    def train(self, state, action, next_state, reward, done, ep):
        self.replay.add((state, action, next_state, reward, done))
        #############
        states, actions, next_states, rewards, dones = self.replay.sample(100) #batch size take from memory
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])
        q_targets = rewards + self.gamma *np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)
 
        #minimum value for epsilon if our training needs more exploration
        if done: self.eps = 0.005+0.995*np.exp(-0.01*ep)

    def __del__(self):
        self.sess.close()


agent = DQNAgent(env)
episodes = 401
Reward = []
Avarage = []

for ep in range(1,episodes):
    
    state= env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
            
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done, ep)
#        env.render() 
        total_reward += reward
        state = next_state

    Reward.append(total_reward)
    
    if ep>9:
        
        Rewardsum = Reward[ep-10] + Reward[ep-9] + Reward[ep-8] + Reward[ep-7] + Reward[ep-6] + Reward[ep-5] + Reward[ep-4] + Reward[ep-3] + Reward[ep-2] + Reward[ep-1]
        Avarage.append(Rewardsum/10)
    print("Episodes:", ep, "total_reward:", total_reward)
#print(type(Reward))

plt.plot(np.linspace(0,episodes,len(Reward),endpoint=False),Reward,color='silver')
plt.plot(np.linspace(10,episodes,len(Avarage),endpoint=False),Avarage)
plt.xlabel('Episode Number')
plt.ylabel('Accumulated Reward')
plt.grid(linestyle='--') 
plt.show() 

