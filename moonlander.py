# %%
from Episode import *
import gymnasium as gym
# %%
env = env = gym.make('LunarLander-v2')
agent = Agent(num_actions = env.action_space.n, num_observations = env.observation_space.shape[0])

# %%
train_episodes = Episode.serie(agent = agent, env = env, nb_ep = 600, isTrain = True)
test_episodes = Episode.serie(agent = agent, env = env, nb_ep = 10, isTrain = True)


# %%
