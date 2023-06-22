import numpy as np
from Agent import Agent
from IPython.display import clear_output
from time import sleep

from gymnasium.wrappers.time_limit import TimeLimit
import time

class Episode:
    def serie(env, agent, nb_ep, isTrain = True, print_info_every = 25, max_nbr_frame = 500):
        start = time.time()
        episodes: list[Episode] = []
        for ep in range(nb_ep):
            episodes.append(Episode(env, agent, isTrain))

            if ep % print_info_every == 0:
                av_scores = np.mean([
                    ep.score for ep in episodes[-print_info_every:]
                ])
                print(
                    f"episode {ep}:",
                    # f"truncated: {episodes[-1].truncated}",
                    # f"done: {episodes[-1].done}",
                    f"average score on last {print_info_every} eps: {av_scores}\n",
                    sep='\n'
                )
                print(f'elapsed {time.time() - start}')

        end = time.time()
        print(f'Serie finished, elapsed {end - start}\n')
        return episodes

    def __init__(self, env: TimeLimit, agent: Agent, isTrain = True):
        self.env = env
        self.isTrain: bool = isTrain
        self.frames: list[Agent.experience] = []
        state, info = self.env.reset()
        
        done, truncated, reward, action = False, False, 0, 0
        
        
        self.frames.append(Agent.experience(state, action, reward, None, done, truncated, False, env.render()))

        while not (done or truncated):
            isExploration, action = agent.action_and_learn(state = state, isTrain = isTrain)
            next_state, reward, done, truncated, info = self.env.step(action)

            frame = Agent.experience(state, action, reward, next_state, done, truncated, isExploration, env.render())
            
            agent.append_experience(frame)
            self.frames.append(frame)
            state = next_state

        
        self.done = done
        self.truncated = truncated
        self.score = np.mean([frame.reward for frame in self.frames])
    
    # def get_hist_df(self): 
    #     return pd.DataFrame(
    #         [(t.state, t.action, t.reward, t.done or t.truncated) for t in self.frames],
    #          columns=['state', 'action', 'reward', "finished"]
    #     )

    def print_frames(self, all = False):
        for i, frame in enumerate(self.frames):
            if not all:
                clear_output(wait=True)
            print(f"Reward: {frame.reward}")
            print(f"State: {frame.state}")
            # print(f"states q_table : {frame.q_table[frame.state]}")
            print(f"Action: {frame.action}")
            print(frame.render)
            print("------------------")
            sleep(.1)

    # def progress_animation(self, all = False):
    #     for frame in self.frames:
    #         if not all:
    #             clear_output(wait=True)
            
    #         title:str = f'state: {frame.state}, action: {frame.action}, reward: {frame.reward}'
    #         fig = frame.q_cm_heatmap(title)
    #         fig.show()
    

    # def CM_path(self):
    #     actions = ""
    #     states = ""
    #     cm_path = np.zeros((self.env.obs_space.n,self.env.obs_space.n))
    #     cm_path_annot = pd.DataFrame([['']*self.env.obs_space.n]*self.env.obs_space.n)

    #     for i in range(len(self.frames)-1):
    #         frame = self.frames[i]
    #         next_frame = self.frames[i+1]

    #         cm_path[frame.state, next_frame.state]+=1
    #         cm_path_annot.loc[frame.state, next_frame.state]+=str(i+1)
    #         actions += str(frame.action) if frame.action != None else ''
    #         states += f"{frame.state}, "
    #     actions += str(next_frame.action)
    #     states += str(next_frame.state)


    #     fig = Tools.heatmap_fig(z = cm_path, hover_text=cm_path, title=states)
    #     fig.show()
