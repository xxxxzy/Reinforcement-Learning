import itertools
import math
import matplotlib.pyplot as plt
import numpy as np


class MDP:
    def reset(self, init_state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class SimpleMDP(MDP):
    def __init__(self, n_states, n_actions, p, r, initial_state_distribution=None):
        self.p = np.asarray(p)
        self.r = np.asarray(r)
        assert self.p.shape == (n_states, n_actions, n_states)
        assert self.r.shape == (n_states, n_actions)

        self.n_states = n_states
        self.n_actions = n_actions
        # Default initial state distribution is uniform
        self.initial_state_distribution = initial_state_distribution or np.ones(self.n_states) / self.n_states

    def reset(self, initial_state=None):
        if initial_state is None:
            self.state = np.random.choice(self.n_states, p=self.initial_state_distribution)
        else:
            self.state = initial_state
        return self.state

    def step(self, action):
        next_state = np.random.choice(self.n_states, p=self.p[self.state, action])
        reward = self.r[self.state, action]
        self.state = next_state
        return next_state, reward



def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank):

    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()

    while sum(p_sa) > 1 + 1e-9:

        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()

    return p_sa


def extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, epsilon):
   
    
    state_value_hat = np.zeros(n_states)
    next_state_value_hat = np.zeros(n_states)
    du = np.zeros(n_states)
    du[0], du[-1] = math.inf, -math.inf

    p_tilde = np.zeros((n_states, n_actions, n_states))
    r_tilde = r_hat + confidence_bound_r
    pi_tilde = np.zeros(n_states, dtype='int')
    while not du.max() - du.min() < epsilon:
        # Sort the states by their values in descending order
        rank = np.argsort(-state_value_hat)
        for st in range(n_states):
            best_ac, best_q = None, -math.inf
            for ac in range(n_actions):
                # print('opt', st, ac)
                # print(state_value_hat)
                # Optimistic transitions
                p_sa_tilde = inner_maximization(p_hat[st, ac], confidence_bound_p[st, ac], rank)
                q_sa = r_tilde[st, ac] + (p_sa_tilde * state_value_hat).sum()
                p_tilde[st, ac] = p_sa_tilde
                if best_q < q_sa:
                    best_q = q_sa
                    best_ac = ac
                    pi_tilde[st] = best_ac
            next_state_value_hat[st] = best_q
            # print(state_value_hat)
        du = next_state_value_hat - state_value_hat
        state_value_hat = next_state_value_hat
        next_state_value_hat = np.zeros(n_states)
        # print('u', state_value_hat, du.max() - du.min(), epsilon)
    return pi_tilde, (p_tilde, r_tilde)


def ucrl2(mdp, delta, initial_state=None):

    
    n_states, n_actions = mdp.n_states, mdp.n_actions
    t = 1
    # Initial state
    st = mdp.reset(initial_state)
    # Model estimates
    total_visitations = np.zeros((n_states, n_actions))
    total_rewards = np.zeros((n_states, n_actions))
    total_transitions = np.zeros((n_states, n_actions, n_states))
    vi = np.zeros((n_states, n_actions))
    for k in itertools.count():
        # Initialize episode k
        t_k = t
        # Per-episode visitations
        vi = np.zeros((n_states, n_actions))
        # MLE estimates
        p_hat = total_transitions / np.clip(total_visitations.reshape((n_states, n_actions, 1)), 1, None)
        # print('p_hat', p_hat)
        r_hat = total_rewards / np.clip(total_visitations, 1, None)
        # print('r_hat', r_hat)

        # Compute near-optimal policy for the optimistic MDP
        confidence_bound_r = np.sqrt(7 * np.log(2 * n_states * n_actions * t_k / delta) / (2 * np.clip(total_visitations, 1, None)))
        confidence_bound_p = np.sqrt(14 * np.log(2 * n_actions * t_k / delta) / np.clip(total_visitations, 1, None))
        # print('cb_p', confidence_bound_p)
        # print('cb_r', confidence_bound_r)
        pi_k, mdp_k = extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
        # print(pi_k, mdp_k)

        # Execute policy
        ac = pi_k[st]
        # End episode when we visit one of the state-action pairs "often enough"
        while vi[st, ac] < max(1, total_visitations[st, ac]):
            next_st, reward = mdp.step(ac)
            # print('step', t, st, ac, next_st, reward)
            yield (t, st, ac, next_st, reward)
            # Update statistics
            vi[st, ac] += 1
            total_rewards[st, ac] += reward
            total_transitions[st, ac, next_st] += 1
            # Next tick
            t += 1
            st = next_st
            ac = pi_k[st]

        total_visitations += vi
#        print(total_rewards)


if __name__ == '__main__':
    
    n_states = 6
    n_actions = 2

    p = [
               [[1, 0, 0, 0, 0, 0],[0.6, 0.4, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0],[0.05, 0.55, 0.4, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0],[0, 0.05, 0.55, 0.4, 0, 0]],
               [[1, 0, 0, 0, 0, 0],[0, 0, 0.05, 0.55, 0.4, 0]],
               [[1, 0, 0, 0, 0, 0],[0, 0, 0, 0.05, 0.55, 0.4]],
               [[1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0.4, 0.6]]
               
            ]
    r = [
            [0.5, 0],[0, 0],[0,0],[0, 0],[0,0],[0,1]
        ]
    """
    p = [
               [ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4, 0]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.55, 0.4,]],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.6]]
               
            ]

#    p = p1+pin+p0
#    print(p)

    r = [
            [0.5, 0],[0, 0],[0, 0],[0,0],[0, 0],[0,0],[0, 0],[0,0],[0, 0],[0,0],[0, 0],[0,0],[0, 0],[0,0],[0,1]
        ]
    """
    mdp = SimpleMDP(n_states, n_actions, p, r)

    transitions = ucrl2(mdp, delta=0.1, initial_state=0)
    tr = []
    r = 0
    for _ in range(250000):
        (t, st, ac, next_st, r) = transitions.__next__()
        tr.append((t, st, ac, next_st, r))

def ucrl2_regret_bound(diameter, n_states, n_actions, delta, t):
    '''Based on Theorem 2 in [JOA10]'''
    return 34 * diameter * n_states * np.sqrt(n_actions * t * np.log(t / delta))

total_regret = []
per_step_regret = []
tt = []
opt_avg_reward = 1
acc_regret = 0
for (t, st, ac, next_st, r) in tr:
    acc_regret += opt_avg_reward - r
    total_regret.append(acc_regret)
    per_step_regret.append(acc_regret / t)
    tt.append(t)
    
total_regret = np.asarray(total_regret)
per_step_regret = np.asarray(per_step_regret)
tt = np.asarray(tt)
diameter_regret_bound = ucrl2_regret_bound(10, 2, 2, 0.1, tt)
#plt.plot(total_regret)
plt.plot(per_step_regret)
#plt.plot(ucrl2_regret_bound(0.1 * 10, 2, 2, 0.1, tt), c='g')
plt.xlabel('Times')

plt.ylabel('Per_step_regret')
plt.show()