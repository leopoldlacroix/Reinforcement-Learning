import seaborn as sns
import numpy as np

class Agent:
    def __init__(self, learning_rate, discount_factor, n_action_space, n_observation_space, state=None,):
        self._q_table:np.ndarray = np.zeros((n_observation_space, n_action_space))
        self.lr = learning_rate
        self.df = discount_factor
        self.current_state = state
        self.last_action = None
    
    def set_state(self, state):
        self.current_state = state

    def action(self):
        possible_action_qs = self._q_table[self.current_state]
        possible_actions = np.arange(possible_action_qs.shape[0])
        best_action_filter = possible_actions[possible_action_qs >= possible_action_qs.max()]

        self.last_action = np.random.choice(possible_actions[best_action_filter])
        return self.last_action
    
    def update(self, reward, new_state):
        self._q_table[self.current_state, self.last_action] = \
            (1-self.lr) * self._q_table[self.current_state, self.last_action] + \
            self.lr * (reward + self.df*self._q_table[new_state].max())
        self.current_state = new_state
    
    def q_table_i(self):
        return sns.heatmap(self._q_table.copy())
