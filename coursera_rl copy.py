# %%
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from Agent import * 
# %%<

ref_state = np.array([[-0.005756  ,  1.402744  , -0.58303285, -0.36339095,  0.00667652,
    0.13206567,  0.        ,  0.        ],])

SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

random.seed(SEED)
tf.random.set_seed(SEED)

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape
num_actions = env.action_space.n

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
def get_action(q_values, epsilon=0.0):

    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(range(4))
    

# %%
agent = Agent(num_actions = env.action_space.n, num_observations = env.observation_space.shape[0])

optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

agent.q_network.load_weights("test.h5")
agent.target_q_network.load_weights("test.h5")

def check_update_conditions(t, num_steps_upd, memory_buffer):

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences

    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + gamma * (1-done_vals)*max_qsa
    
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    loss = tf.reduce_mean((q_values - y_targets)**2)
    
    return loss


def update_target_network(q_network, target_q_network):

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network):

    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    update_target_network(q_network, target_q_network)

def get_new_epsilon(epsilon):

    return max(E_MIN, E_DECAY * epsilon)


def get_action(q_values, epsilon=0.0):

    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(range(4))

start = time.time()

num_episodes = 600
max_num_timesteps = 1000

total_point_history = []
total_frames_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

memory_buffer = deque(maxlen=MEMORY_SIZE)

for i in range(num_episodes):
    
    state, _ = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        is_exp, action = agent.action(state)
        
        next_state, reward, done, truncated, info = env.step(action)
        
        # memory_buffer.append(experience(state, action, reward, next_state, done))
        agent.experience_memory.append(experience(state, action, reward, next_state, done))


        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, agent.experience_memory)
        # update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        # assert update_a == update
        # assert memory_buffer == agent.experience_memory

        if update:
            experiences = Agent.get_experiences(agent.experience_memory)
            # experiences = get_experiences(memory_buffer)
            
            agent.agent_learn(experiences, GAMMA)
            # agent.agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward
        
        if done:    
            break
            
    total_point_history.append(total_points)
    total_frames_history.append(t)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    av_frame_count = np.mean(total_frames_history[-num_p_av:])
    
    agent.update_epsilon()

    ref_pred: tf.Tensor = agent.q_network(ref_state)
    ref_pred = "("+",".join(ref_pred.numpy()[0].astype(dtype=str))+")"
    print(f"\rEpisode {i+1} | last {num_p_av} episodes: {av_latest_points=:f} ,{av_frame_count=:.2f} ,{agent.epsilon=:2f} ,{ref_pred}", end = "")

    if (i+1) % num_p_av == 0:
        print(f"Episode {i+1} | last {num_p_av} episodes: {av_latest_points=:f} ,{av_frame_count=:.2f} ,{agent.epsilon=:2f},{ref_pred}")

    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
# %%
