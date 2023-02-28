# %%
from Episode import *
from EnvManager import *


env_name = "Taxi-v3"
env_manager = EnvManager(env_name, max_episode_steps = 200) 
agent = Agent(0.1, 0.7, 0.2, env_manager.action_space, env_manager.obs_space)


# %%
# training
train_episodes = Episode.serie(agent=agent, env_manager=env_manager,nb_ep=100000, isTrain=True)
test_episodes = Episode.serie(agent=agent, env_manager=env_manager,nb_ep=100, isTrain=True)
