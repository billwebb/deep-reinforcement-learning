import numpy as np
from collections import defaultdict

class Agent:

    episode = 1

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.a = 0.2
        self.g = 0.9
        self.e = 1.0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() > self.e:
            a = np.argmax(self.Q[state])
        else:
            a = np.random.randint(self.nA)
        return a

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.a * (reward + (self.g * \
            (np.amax(self.Q[next_state]) - self.Q[state][action])))
        if done:
            self.e = 1 / self.episode
            self.episode += 1
