# %%
from Episode import *
from EnvManager import *
# %%
env_name = "FrozenLake-v1"
env_manager = EnvManager(env_name) 
agent = Agent(0.5, 0.5, env_manager.n_action_space, env_manager.n_obs_space)


# %%
episodes: list[Episode] = []
for ep in range(20):
    episodes.append(Episode(env_manager, agent))
    if ep % 20 == 0:
        print(
            f"episode {ep}:",
            f"truncated: {episodes[-1].truncated}",
            f"done: {episodes[-1].done}",
            sep='\n'
        )


episodes[-1].CM_path()

# %%
for ep in episodes:
    plt.figure()
    ep.frames[-1].q_cm_heatmap()
    plt.show()
# %%
