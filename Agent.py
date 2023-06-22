# %%
from collections import namedtuple, deque
import numpy as np

from gymnasium.spaces.discrete import Discrete
import tensorflow as tf
# %%
class Agent:

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "truncated", "isExploration", "render"])
    """fields: state, action, reward, next_state, done, truncated, isExploration, render"""
    min_batch_size = 64
    
    def __init__(
            self,
            num_actions: int,
            num_observations: int,
            learning_rate = 1e-3,
            t_soft_update = 0.8,  
            discount_factor = 0.995,
            num_steps_for_update = 4,
            memory_size = 100_000
        ):
        self.epsilon = 1.00
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.t_soft_update = t_soft_update
        self.num_steps_for_update = num_steps_for_update
        self.experience_memory:deque[Agent.experience] = deque(maxlen = memory_size)
        
        self.num_actions = num_actions
        self.action_counter = 0

        # Create the Q-Network
        self.q_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_observations)),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=num_actions, activation="linear"), 
            ])

        # Create the target Q^-Network
        self.target_q_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_observations)),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=num_actions, activation="linear"),
            ])
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    
    def action_and_learn(self, state: np.ndarray, isTrain: bool)-> tuple[bool, int]:
        self.action_counter +=1

        do_training =  len(self.experience_memory) > Agent.min_batch_size and isTrain and self.action_counter > 4
        if do_training:
            self.agent_learn()


        if np.random.random() < self.epsilon:
            return True, np.random.randint(self.num_actions)
        
               
        q_values = self.q_network(state)
        return False, np.argmax(q_values)
            
    def append_experience(self, exp: experience):
        self.experience_memory.append(exp)

    def compute_loss(self):

        
        # Unpack the mini-batch of experience tuples
        states, actions, rewards, next_states, dones, truncateds, isExplorations, renders = map(
            lambda l: np.array([*l]),
            np.array(self.experience_memory, dtype=object).T
        )
        
        
        max_qsa = tf.reduce_max(self.target_q_network(next_states), axis=-1)
        y_targets = rewards + self.discount_factor * (1-dones)*max_qsa    
        
        q_values = self.q_network(states)
        action_range_action_stack = tf.stack([tf.range(q_values.shape[0]),tf.cast(actions, tf.int32)], axis=1)
        
        q_values = tf.gather_nd(q_values, action_range_action_stack)
        loss = tf.reduce_mean((q_values - y_targets)**2)
        
        return loss

    def _target_network_soft_update(self, q_network: tf.keras.Sequential, target_q_network: tf.keras.Sequential):
        for target_weights, q_net_weights in zip(
            target_q_network.weights, q_network.weights
        ):
            target_weights.assign(self.t_soft_update * q_net_weights + (1.0 - self.t_soft_update) * target_weights)
        
    @tf.function
    def agent_learn(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Update the weights of the q_network.
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # update the weights of target q_network
        self._target_network_soft_update(self.q_network, self.target_q_network)
