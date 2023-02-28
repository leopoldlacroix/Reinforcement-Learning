import plotly.express as px
import numpy as np
from gym.spaces import Space
class Agent:
    def __init__(self, learning_rate, discount_factor, exploration_rate, action_space:Space, observation_space: Space, state=None):
        self._q_table:np.ndarray = np.zeros((observation_space.n, action_space.n))
        self.lr = learning_rate
        self.er = exploration_rate
        self.df = discount_factor
        self.current_state = state
        self.last_action = None
        self.action_space = action_space
    
    def set_state(self, state):
        self.current_state = state

    """ Takes action, returns tuple(bool, int) is exploration/action """
    def action(self, explore = True)-> tuple[bool, int]:
        possible_action_qs = self._q_table[self.current_state]
        possible_actions = np.arange(possible_action_qs.shape[0])

        if(explore and np.random.uniform(0,1)<self.er):
            return (True, self.action_space.sample())

        best_action_filter = possible_actions[possible_action_qs >= possible_action_qs.max()]

        self.last_action = np.random.choice(possible_actions[best_action_filter])
        return (False, self.last_action)
    
    def update(self, reward, new_state):
        self._q_table[self.current_state, self.last_action] = \
            (1-self.lr) * self._q_table[self.current_state, self.last_action] + \
            self.lr * (reward + self.df*self._q_table[new_state].max())
        self.current_state = new_state
    
    def q_table_i(self):
        return px.imshow(self._q_table.copy())
