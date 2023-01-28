
# %%
import gym
from time import sleep
from IPython.display import clear_output
import numpy as np

# %%
env_name = "FrozenLake-v1"
# %%

class Agent:
    def __init__(self, learning_rate, discount_factor, n_action_space, n_observation_space, state=None,):
        self._q_table:np.ndarray = np.zeros((n_observation_space, n_action_space))+1
        self.lr = learning_rate
        self.df = discount_factor
        self.current_state = state
        self.last_action = None
    
    def set_state(self, state):
        self.current_state = state

    def action(self):
        self.last_action = self._q_table[self.current_state].argsort()[-1]
        return self.last_action
    
    def update(self, reward, new_state):
        self._q_table[self.current_state, self.last_action] = \
            (1-self.lr) * self._q_table[self.current_state, self.last_action] + \
            self.lr * (reward + self.df*self._q_table[new_state].max())
        self.current_state = new_state
    
    def q_table_i(self):
        return self._q_table.copy()

class Frame:
    def __init__(self, representation, state, action, reward, q_table):
        self.representation = representation
        self.state = state
        self.action = action
        self.reward = reward
        self.q_table = q_table
class Episode:
    def __init__(self, frames, done, truncated):
        self.frames = frames
        self.done = done
        self.truncated = truncated

def episode(env: gym.Env, agent:Agent) -> Episode:
    state, info = env.reset()
    agent.set_state(state)

    actions_taken = 0
    frames = []
    penalties=0
    done, truncated  = False, False
    while not (done or truncated):
        action = agent.action()
        state, reward, done, truncated, info = env.step(action)
        agent.update(reward, state)

        frames.append(
            Frame(
                representation = env.render(),
                state = state,
                action = action,
                reward = reward,
                q_table = agent.q_table_i()
            )
        )
        actions_taken +=1

    return Episode(frames = frames, done = done, truncated = truncated)

def print_frames(frames: list[Frame]):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame.representation)
        print(f"Timestep: {i + 1}")
        print(f"State: {frame.state}")
        print(f"Action: {frame.action}")
        print(f"Reward: {frame.reward}")
        print(f"\n q_table: {frame.q_table}")
        input()
    
# %%

env : gym.Env = gym.make(env_name, render_mode="ansi", max_episode_steps = 2000)

episodes: list[Episode] = []
agent = Agent(0.5, 0.5, env.action_space.n, env.observation_space.n)
for ep in range(100):
    episodes.append(episode(env, agent))
    if ep % 20 == 0:
        print(
            f"episode {ep}:",
            f"truncated: {episodes[-1].truncated}",
            f"done: {episodes[-1].done}",
            sep='\n'
        )


# %%
for ep in episodes:
    print_frames(ep.frames)
# %%
