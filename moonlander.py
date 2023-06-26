# %%
from Episode import *
import gymnasium as gym
import random

random.seed(0)
# %%
env: gym.Env = gym.make('LunarLander-v2')
agent = Agent(num_actions = env.action_space.n, num_observations = env.observation_space.shape[0])

# %%
train_episodes = Episode.serie(agent = agent, env = env, nb_ep = 600, is_train = True, print_info_every=100)
test_episodes = Episode.serie(agent = agent, env = env, nb_ep = 10, is_train = True)

# %%
