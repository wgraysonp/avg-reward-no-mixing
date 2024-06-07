from __future__ import division
import numpy as np
from collections import defaultdict


class StochasticApproximationAgent(object):
    """
    This agent implements the standard Q-learning algorithm for the infinite-horizon
    average-reward setting with epsilon-greedy exploration.
    """
    def __init__(self, env, epsilon):
        self.REF_STATE = 0
        self.alpha = 1.
        self.env = env
        self.epsilon = epsilon

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])  # for consistency called self.mu (it is q).
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.state = None
        self.action = None
        self.t = 1

        self.reset()

    def act(self, state):
        self.state = state

        if np.random.rand() <= self.epsilon:
            self.action = np.random.choice(self.env.actions)
        else:
            self.action = np.argmax(self.mu[self.state])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.alpha = 1.0/self.n[self.state, self.action]
        self.n_prime[self.state, self.action, next_state] += 1
        self.mu[self.state, self.action] = (1 - self.alpha) * self.mu[self.state, self.action] \
                                          + self.alpha * (reward + np.max(self.mu[next_state]) - np.max(self.mu[self.REF_STATE]))
        self.t += 1

    def info(self):
        return 'name = StochasticApproximationAgent\n' + 'epsilon = {0}\n'.format(self.epsilon)

    def reset(self):
        self.state = None
        self.action = None

        self.mu = np.zeros([self.env.nb_states, self.env.nb_actions])
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.n_prime = np.zeros([self.env.nb_states, self.env.nb_actions, self.env.nb_states], dtype=int)

        self.alpha = 1.
        self.t = 1


class RMDAgent(object):
    """
    Implements the recurrent mirror descent algorithm from our paper

    Attributes:
        eta: step size
        s_bar: atom state
        N: Number of returns to s_bar
        B: cuttof threshold for Q function evaluation
        T: Max number of samples for estimating J
        env: Learning envirionment
        state: agent's current state
        action: agent's action
        policy: current policy
        t: counter for number of samples
        C_bar: counter of current returns to s_bar
        q_trajectory: collected trajectory for estimating q during an episode
        j_trajectory: collected trajectory for estimateing J during an episode
        j_samples: geometric random number of samples to use to estimate J
        j_collect: True if at the current time we are collecting samples to estimate J
        q_collect: True if at the current time we are collecting samples to estimate Q
    """

    def __init__(self, env, s_bar, N, B, T, eta=0.01):

        self.eta = eta
        self.s_bar = s_bar
        self.C_bar = 0
        self.N = N
        self.B = B
        self.T = T

        self.env = env
        self.state = None
        self.action = None

        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.t = 1

        self.q_trajectory = []
        self.j_trajectory = []
        self.count_j_samples = 0
        self.j_samples = 2**np.random.geometric(p=0.5)
        if self.j_samples > self.T:
            self.j_samples = 1

        self.j_collect = True
        self.q_collect = False

    def act(self, state):
        self.state = state
        if self.q_collect and self.C_bar == self.N:
            self.C_bar = 0
            self.q_collect = False
            self.j_collect = True
            self._rmd_update()
        self.action = np.random.choice(self.env.actions, p=self.policy[state])
        raise NotImplementedError

    def update(self, next_state, reward):
        assert not (self.j_collect and self.q_collect), "Error: j_collect and q_collect are both true"
        if self.q_collect:
            self.q_trajectory.append((self.state, self.action, reward))
            if next_state == self.s_bar:
                self.C_bar += 1
        elif self.j_collect:
            self.j_trajectory.append((self.state, self.action, reward))
            self.count_j_samples += 1
            if self.count_j_samples == 2**self.j_samples:
                self.count_j_samples = 0
                self.j_collect = False
                self.j_samples = 2**np.random.geometric(p=0.5)
                if self.j_samples > self.T:
                    self.j_samples = 1
        else:
            "We have finished sampling J and are waiting to return to s_bar to start collecting samples for Q"
            if next_state == self.s_bar:
                self.q_collect = True

    def info(self):
        return 'name = RMDAgent\n' + 'N = {0}\n'.format(self.N) + 'T = {0}\n'.format(self.T)\
                    + 'B = {0}\n'.format(self.B)

    def reset(self):
        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.q_trajectory = []
        self.j_trajectory = []
        self.count_j_samples = 0
        self.j_samples = 2**np.random.geometric(p=0.5)
        if self.j_samples > self.T:
            self.j_samples = 1
        self.C_bar = 0
        self.t = 1


    def _rmd_update(self):
        q_hat = self._estimate_q()
        raise NotImplementedError

    def _estimate_q(self):
        assert self.q_trajectory[0][0] == self.s_bar, "initial state in q trajectory is not s_bar"

        cum_reward = np.cumsum([item[2] for item in self.q_trajectory])

        d = defaultdict(list)
        for i, item in enumerate(self.q_trajectory):
            d[item[0]].append(i)

        j_hat = self._estimate_j()

        q_hat = np.zeros([self.env.nb_states, self.env.nb_actions])
        for s in self.env.states:
            Z = np.zeros(self.env.nb_actions)
            i = 0
            tau = 0
            for j in d[s]:
                try:
                    end = np.min([x for x in d[self.s_bar] if x > j]) - 1
                except ValueError:
                    end = len(self.q_trajectory) - 1
                R = cum_reward[end] - cum_reward[j-1] - (end - j + 1)*j_hat
                i+=1
                Z[self.q_trajectory[j][1]] += float(R)/self.policy[s, self.q_trajectory[j][1]]
            if i > 0:
                for a in self.env.actions:
                    if np.abs(self.policy[s, a]*Z[a]/i) > self.B:
                        Z[a] = 0
                q_hat[s, :] = Z/i

        self.q_trajectory = []
        return q_hat

    def _estimate_j(self):
        m = np.log2(len(self.j_trajectory))
        if m == 0:
            j_hat = self.j_trajectory[0][2]
        else:
            s_1 = np.mean([item[2] for item in self.j_trajectory])
            s_2 = np.mean([item[2] for item in self.j_trajectory[:2**(m-1)]])
            j_hat = self.j_trajectory[0][2] + 2**m(s_1 - s_2)

        self.j_trajectory = []

        return j_hat


class OptimisticDiscountedAgent(object):
    """
    Implements the Optimistic Q-learning algorithm from Wei et al 2020
    """

    def __init__(self, env, gamma=0.99, c=1.0):
        # _________ constants __________
        self.gamma = gamma
        self.H = gamma/(1.0-gamma)
        self.c = c
        # ______________________________

        self.env = env
        self.state = None
        self.action = None

        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])  # Q_hat in the algorithm
        self.v_hat = self.H * np.ones(self.env.nb_states)

    def act(self, state):
        self.state = state
        self.action = np.argmax(self.mu_hat[self.state, :])
        return self.action

    def update(self, next_state, reward):
        self.n[self.state, self.action] += 1
        self.t += 1
        bonus = self.c * np.sqrt(self.H/self.n[self.state, self.action])
        alpha = (self.H + 1)/(self.H + self.n[self.state, self.action])
        self.mu[self.state, self.action] = (1-alpha)*self.mu[self.state, self.action] + alpha*(reward + self.gamma*self.v_hat[next_state] + bonus)
        self.mu_hat[self.state, self.action] = min(self.mu_hat[self.state, self.action], self.mu[self.state, self.action])
        self.v_hat[self.state] = np.max(self.mu_hat[self.state, :])

    def info(self):
        return 'name = OptimisticDiscountedAgent\n' + 'gamma = {0}\n'.format(self.gamma) + 'c = {0}\n'.format(self.c)

    def reset(self):
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
        self.mu = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.mu_hat = self.H * np.ones([self.env.nb_states, self.env.nb_actions])
        self.v_hat = self.H * np.ones(self.env.nb_states)


class OOMDAgent(object):
    """
    Implements the MDP-OOMD algorithm from Wei et al 2020.
    """
    def __init__(self, env, N, B, eta=0.01):
        # ____________constants____________
        self.eta = eta
        self.N = N
        self.B = B
        if self.B < self.N:
            raise ValueError('B should be larger than N')
        # _________________________________
        self.env = env
        self.state = None
        self.action = None

        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.policy_prime = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.episode_trajectory = []  # each element of this list is a tuple of (state, action, reward)
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)

    def act(self, state):
        self.state = state
        if self.t % self.B == 1: # a new episode starts
            self._oomd_update()
        self.action = np.random.choice(self.env.actions, p=self.policy[state])
        return self.action

    def update(self, next_state, reward):
        self.episode_trajectory.append((self.state, self.action, reward))
        self.n[self.state, self.action] += 1
        self.t += 1

    def info(self):
        return 'name = OOMDAgent\n' + 'N = {0}\n'.format(self.N) + 'B = {0}\n'.format(self.B)

    def reset(self):
        self.policy = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.policy_prime = (1.0/self.env.nb_actions)*np.ones((self.env.nb_states, self.env.nb_actions))
        self.episode_trajectory = []
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)

    def _oomd_update(self):

        #  pre-processing the trajectory for fast access
        cum_reward = np.cumsum([item[2] for item in self.episode_trajectory])
        d = defaultdict(list)  # dictionary of {state: [indices]} time indexes of visiting that state
        for i, item in enumerate(self.episode_trajectory):
            d[item[0]].append(i)

        #  estimate q
        beta_hat = np.zeros([self.env.nb_states, self.env.nb_actions])
        for s in self.env.states:
            y = np.zeros(self.env.nb_actions)
            tau = 0
            i = 0
            for j in d[s]:
                if self.B - self.N > j >= tau:
                    R = cum_reward[j + self.N - 1] - cum_reward[j - 1] if j >= 1 else cum_reward[j + self.N - 1]
                    # tau = j + 2 * self.N
                    tau = j+1
                    i += 1
                    y[self.episode_trajectory[j][1]] += float(R)/self.policy[s, self.episode_trajectory[j][1]]
            if i > 0:
                beta_hat[s] = y/i
        self.episode_trajectory = []

        # online mirror descent update
        for s in self.env.states:
            lamda_prime = self._binary_search(self.policy_prime[s], self.eta*beta_hat[s].min(), self.eta*beta_hat[s].max(), beta_hat[s])
            self.policy_prime[s] = self._update_policy(self.policy_prime[s], lamda_prime, beta_hat[s])

            lamda = self._binary_search(self.policy[s], self.eta*beta_hat[s].min(), self.eta*beta_hat[s].max(), beta_hat[s])
            self.policy[s] = self._update_policy(self.policy_prime[s], lamda, beta_hat[s])
            self.policy[s] /= self.policy[s].sum()

    def _binary_search(self, x, low, high, q_a_array):
        tol = 0.0005
        while True:
            lamda = (low+high)/2.0
            y_updated = self._update_policy(x, lamda, q_a_array)
            if abs(high-low) < tol:
                return high
            if (y_updated < 0).any():
                low = lamda
            elif sum(y_updated) < 1 - tol:
                high = lamda
            elif sum(y_updated) > 1 + tol:
                low = lamda
            else:
                return lamda

    def _update_policy(self, pi, lamda, q_a_array):
        new_pi = np.zeros(self.env.nb_actions)
        for a in range(self.env.nb_actions):
            new_pi_reci = 1.0/pi[a] - self.eta*q_a_array[a] + lamda
            new_pi[a] = 1.0/new_pi_reci
        return new_pi



class PolitexAgent(object):
    """
    Implements the Politex algorithm from Abbasi-Yadkori 2019
    """

    def __init__(self, env, N, B, eta=0.2):
        # _________ constants __________
        self.eta = eta
        self.N = N
        self.B = B
        # ______________________________

        self.env = env
        self.state = None
        self.action = None

        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # total number of visits to (s, a)

        self.ns = np.zeros([self.env.nb_states], dtype=int)  # number of visits to s in phase N of each episode
        self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)  # number of visits to ss' in phase N of each episode
        self.policy = np.ones([self.env.nb_states, self.env.nb_actions]) * 1.0/self.env.nb_actions

        self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at second phase of each episode
        self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at each episode
        self.q = np.zeros([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.v = np.zeros([self.env.nb_states])  # v_pi of the algorithm
        self.lamda = 0

    def act(self, state):
        self.state = state
        if self.t % self.B <= self.N: 
            self.action = np.random.choice(self.env.actions, p=self.policy[self.state])
        else:
            for a in self.env.actions:
                if self.n_second_phase[self.state, a]==0:
                   self.action = a
                   return self.action
            self.action = np.random.choice(self.env.actions, p=self.policy[self.state])
        return self.action

    def update(self, next_state, reward):
        if self.t % self.B == self.N:  # beginning of second phase of the episode
            self.lamda = 1.0/self.N * np.sum([self.n_episode[s, a] * self.env.rewards[s, a] for s in self.env.states for a in self.env.actions])
            mat = np.diag(self.ns) - self.nss_prime
            vec = np.sum(self.n_episode * (self.env.rewards - self.lamda), axis=1)
            self.v = np.dot(np.linalg.pinv(mat + 0.05 * np.eye(self.env.nb_states)), vec)
        if self.t % self.B == 0:  # beginning of an episode
            for s in self.env.states:
                for a in self.env.actions:
                    if self.n_second_phase[s, a] > 0:
                        self.q[s,a] /= self.n_second_phase[s, a]
            self.policy = self.policy * np.exp(self.eta * np.maximum(np.minimum(self.q, 100/self.eta),-100/self.eta))
            for s in self.env.states:
                self.policy[s] /= np.sum(self.policy[s])
                #self.policy[s] = 0.999*self.policy[s] + 0.001*np.ones((self.env.nb_actions,))/self.env.nb_actions
                #self.policy[s] /= np.sum(self.policy[s])
                
            self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
            self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)
            self.ns = np.zeros([self.env.nb_states], dtype=int)
            self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)
            self.q = np.zeros([self.env.nb_states, self.env.nb_actions])

        if self.t % self.B > self.N:  # second phase of an episode
            self.n_second_phase[self.state, self.action] += 1
            self.q[self.state, self.action] += self.env.rewards[self.state, self.action] - self.lamda + self.v[next_state]
        self.n[self.state, self.action] += 1
        self.ns[self.state] += 1
        self.nss_prime[self.state, next_state] += 1
        self.n_episode[self.state, self.action] += 1
        self.t += 1

    def info(self):
        return 'name = PolitexAgent\n' + 'N = {0}\n'.format(self.N) + 'B = {0}\n'.format(self.B) + 'eta = {0}\n'.format(self.eta)

    def reset(self):
        self.t = 1
        self.n = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # total number of visits to (s, a)

        self.ns = np.zeros([self.env.nb_states], dtype=int)  # number of visits to s in phase N of each episode
        self.nss_prime = np.zeros([self.env.nb_states, self.env.nb_states], dtype=int)  # number of visits to ss' in phase N of each episode
        self.policy = np.ones([self.env.nb_states, self.env.nb_actions]) * 1.0/self.env.nb_actions

        self.n_second_phase = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at second phase of each episode
        self.n_episode = np.zeros([self.env.nb_states, self.env.nb_actions], dtype=int)  # number of visits to (s, a) at each episode
        self.q = np.zeros([self.env.nb_states, self.env.nb_actions])  # Q in the algorithm
        self.v = np.zeros([self.env.nb_states])  # v_pi of the algorithm
        self.lamda = 0


