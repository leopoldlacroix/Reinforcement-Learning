import plotly.express as px
import numpy as np
from gym.spaces import Space
from tensorflow.keras import Sequetial
from tensorflow.keras.leayers import Input, Dense
class Agent:
    def __init__(
            self,
            action_space:Space,
            observation_space: Space,
            learning_rate = 1e-3,  
            discount_factor = 0.995,
            num_steps_for_update = 4,
            memory_size = 100_000,
            state=None
        ):
        self.learning_rate = learning_rate
        self.num_steps_for_update = num_steps_for_update
        self.discount_factor = discount_factor
        self.current_state = state
        self.last_action = None
        self.action_space = action_space
        self.memory_size = memory_size

        # Create the Q-Network
        q_network = tf. Sequential([
            ### START CODE HERE ### 
            Input(shape=(observation_space.n)),
            Dense(units=64, activation="relu"),
            Dense(units=64, activation="relu"),
            Dense(units=action_space.n, activation="linear"),
            
            ### END CODE HERE ### 
            ])

        # Create the target Q^-Network
        target_q_network = Sequential([
            ### START CODE HERE ### 
            Input(shape=(observation_space.n)),
            Dense(units=64, activation="relu"),
            Dense(units=64, activation="relu"),
            Dense(units=action_space.n, activation="linear"),
        
            ### END CODE HERE ###
            ])
    
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
