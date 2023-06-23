import numpy as np
from Agent import Agent
from IPython.display import clear_output
from time import sleep
from collections import deque

from gymnasium.wrappers.time_limit import TimeLimit
import time

class Episode:
    def serie(env, agent: Agent, nb_ep, isTrain = True, print_info_every = 100, max_nbr_frame = 1000):
        start = time.time()
        scores: deque[float] = deque(maxlen = 1000)
        frame_counts: deque[float] = deque(maxlen = 1000)
        for ep_i in range(nb_ep):
            ep_o = Episode(env, agent, isTrain, max_nbr_frame)
            scores.append(ep_o.score)
            frame_counts.append(ep_o.frame_count)

            av_frames = np.mean(list(frame_counts)[-print_info_every:])
            av_scores = np.mean(list(scores)[-print_info_every:])
            print(f"\rEpisode {ep_i} | last {print_info_every} episodes (av_score: {av_scores:.2f}, av_frames: {av_frames:.2f}", end = "")
            if (ep_i) % print_info_every == 0:
                print(f"\rEpisode {ep_i} | last {print_info_every} episodes (av_score: {av_scores:.2f}, av_frames: {av_frames:.2f}")
                print(f'elapsed {time.time() - start}')
            
            agent.update_epsilon()

        end = time.time()
        print(f'Serie finished, elapsed {end - start}\n')
        return scores
    
    def __init__(self, env: TimeLimit, agent: Agent, isTrain = True, max_nbr_frame = 1000: int):
        self.env = env
        # self.isTrain: bool = isTrain
        # self.frames: list[Agent.experience] = []
        rewards: list[float] = []
        state, info = self.env.reset()
        
        done, truncated, reward, action = False, False, 0, 0
        
        # self.frames.append(Agent.experience(state, action, reward, None, done, truncated, False, env.render()))
        
        for frame_i in range(max_nbr_frame):
            if done or truncated:
                break
            isExploration, action = agent.action_and_learn(state = state, isTrain = isTrain)
            next_state, reward, done, truncated, info = self.env.step(action)

            frame = Agent.experience(state, action, reward, next_state, done, truncated, isExploration, None)
            # frame = Agent.experience(state, action, reward, next_state, done, truncated, isExploration, env.render())
            
            agent.append_experience(frame)
            # self.frames.append(frame)
            rewards.append(reward)
            state = next_state

        self.frame_count = frame_i
        self.done = done
        self.truncated = truncated
        # self.score = np.mean([frame.reward for frame in self.frames])
        self.score = np.sum(rewards)
    
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
