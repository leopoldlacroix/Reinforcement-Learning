# %%
from collections import namedtuple, deque
import numpy as np

from gymnasium.spaces.discrete import Discrete
import tensorflow as tf
import random
# %%
ref_state = np.array([-0.005756  ,  1.402744  , -0.58303285, -0.36339095,  0.00667652,
    0.13206567,  0.        ,  0.        ])
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
            memory_size = 1_000,
            update_network_every = 4,
            update_epsilon_every = 60,
            e_decay = 0.995,
            e_min = 0.01
        ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.t_soft_update = t_soft_update
        self.e_decay = e_decay
        self.e_min = e_min
        self.update_network_every = update_network_every
        self.update_epsilon_every = update_epsilon_every

        self.network_update_counter = 0
        self.epsilon_update_counter = 1
        self.experience_memory:deque[Agent.experience] = deque(maxlen = memory_size)
        self.epsilon = 1

        self.num_actions = num_actions

        # Create the Q-Network
        self.q_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_observations)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation="linear"), 
            ])

        # Create the target Q^-Network
        self.target_q_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_observations)),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=num_actions, activation="linear"),
            ])
        
        self.target_q_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def update_epsilon(self):
        # print(self.epsilon)
        self.epsilon = max(self.e_min, self.e_decay* self.epsilon)
    
    def action_and_learn(self, state: np.ndarray, isTrain: bool)-> tuple[bool, int]:
        self.network_update_counter +=1

        do_training = (self.network_update_counter%self.update_network_every == 0) and len(self.experience_memory) > Agent.min_batch_size and isTrain
        if do_training:
            # self.epsilon_update_counter +=1
            experiences = random.sample(self.experience_memory, k=64)
            states, actions, rewards, next_states, dones, truncateds, isExplorations, renders = map(
                lambda l: tf.convert_to_tensor(np.array([*l]), dtype = tf.float32),
                np.array(experiences, dtype=object).T
            )
            experiences = (states, actions, rewards, next_states, dones)
            self.agent_learn(experiences, self.discount_factor)

        # if self.epsilon_update_counter%self.update_epsilon_every == 0:
        #     # print('epsilon')
        #     self.epsilon_update_counter=1
        #     self.update_epsilon()

        if np.random.random() < self.epsilon:
            return True, np.random.randint(self.num_actions)
        
               
        q_values = self.q_network(np.array([state,]))
        return False, np.argmax(q_values)
    

    def append_experience(self, exp: experience):
        self.experience_memory.append(exp)

    def compute_loss(self, experiences, gamma, q_network, target_q_network):
        """gamma = discount factor"""
        # Unpack the mini-batch of experience tuples
        states, actions, rewards, next_states, dones = experiences
        max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
        y_targets = rewards + gamma * (1-dones)*max_qsa    
        
        q_values = q_network(states)
        action_range_action_stack = tf.stack([tf.range(q_values.shape[0]),tf.cast(actions, tf.int32)], axis=1)
        
        q_values = tf.gather_nd(q_values, action_range_action_stack)
        loss = tf.reduce_mean((q_values - y_targets)**2)
        
        return loss

    def _target_network_soft_update(self):
        for target_weights, q_net_weights in zip(
            self.target_q_network.weights, self.q_network.weights
        ):
            target_weights.assign(self.t_soft_update * q_net_weights + (1.0 - self.t_soft_update) * target_weights)
        
    @tf.function
    def agent_learn(self, experiences, gamma):
        # print("Updating network")
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, gamma, self.q_network, self.target_q_network)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Update the weights of the q_network.
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # update the weights of target q_network
        self._target_network_soft_update()

    def __repr__(self) -> str:
        return self.__