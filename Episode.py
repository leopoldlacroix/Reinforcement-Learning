from Agent import Agent
from IPython.display import clear_output
from time import sleep

from EnvManager import *


class Frame:
    def __init__(self, env_manager: EnvManager, state, action, reward, q_table, done, truncated, isExploration):
        self.env_manager = env_manager
        self.rep = env_manager.render()
        self.state = state
        self.action = action
        self.reward = reward
        self.q_table = q_table
        self.done = done
        self.truncated = truncated
        self.isExploration = isExploration
        """Adjency matrix with Q values as connection weight"""
        # self.q_CM = self.weigthed_connection_matrix_from_q_table()
    
    def weigthed_connection_matrix_from_q_table(self):
        lc = self.env_manager.obs_space.n
        connection_matrix: np.ndarray = np.zeros((lc, lc))
        for state in self.env_manager.P:
            for action in list(self.env_manager.P[state].keys())[::-1]:
                for p, next_state, reward, done in self.env_manager.P[state][action]:
                    connection_matrix[state, next_state] = self.q_table[state, action]

        return connection_matrix
    
    def q_cm_heatmap(self) -> Figure:
        title = f'state: {self.state}, action: {self.action}, reward: {self.reward}'
        return Tools.heatmap_fig(z = self.q_CM, hover_text=self.env_manager.hover_matrix, title = title)
            

    def __repr__(self) -> str:
        return f"state: {self.state}, action: {self.action}, reward: {self.reward}, truncated: {self.truncated}, done: {self.done}"
    
import time

class Episode:
    def serie(env_manager, agent, nb_ep, isTrain = True):
        start = time.time()
        episodes: list[Episode] = []
        for ep in range(nb_ep):
            episodes.append(Episode(env_manager, agent, isTrain))
            if ep % 200 == 0:
                print(
                    f"episode {ep}:",
                    f"truncated: {episodes[-1].truncated}",
                    f"done: {episodes[-1].done}",
                    sep='\n'
                )
                print(f'elapsed {time.time() - start}')
        end = time.time()
        print(f'elapsed {end - start}')
        return episodes

    def __init__(self, env_manager: EnvManager, agent: Agent, isTrain=True):
        self.env_manager = env_manager
        self.isTrain = isTrain
        self.frames:list[Frame] = []
        state, info = self.env_manager.reset()
        agent.set_state(state)
        
        done, truncated, reward, action = False, False, None, None
        

        self.add_frame(self.env_manager, state, action, reward, agent._q_table, done, truncated, False)
        while not (done or truncated):
            isExploration, action = agent.action(explore=isTrain)
            state, reward, done, truncated, info = self.env_manager.step(action)
            reward = -1 if done and reward == 0 else reward

            agent.update(reward, state)         
            self.add_frame(self.env_manager, state, action, reward, agent._q_table, done, truncated, isExploration)

            
        agent.update(reward, state)
        self.done = done
        self.truncated = truncated
    
    def get_hist_df(self): 
        return pd.DataFrame(
            [(t.state, t.action, t.reward, t.done or t.truncated) for t in self.frames],
             columns=['state', 'action', 'reward', "finished"]
        )


    
    def add_frame(self, env_manager, state, action, reward, q_table, done, truncated, isExploration):
        self.frames.append(
            Frame(
            env_manager= env_manager,
                state = state,
                action = action,
                reward = reward,
                q_table = q_table,
                done = done, 
                truncated = truncated,
                isExploration = isExploration
            )
        )

    def print_frames(self, all = False):
        for i, frame in enumerate(self.frames):
            if not all:
                clear_output(wait=True)
            print(f"Reward: {frame.reward}")
            print(f"State: {frame.state}")
            print(f"states q_table : {frame.q_table[frame.state]}")
            print(f"Action: {frame.action}")
            print(frame.rep)
            print("------------------")
            sleep(.1)

    def progress_animation(self, all = False):
        for frame in self.frames:
            if not all:
                clear_output(wait=True)
            
            title:str = f'state: {frame.state}, action: {frame.action}, reward: {frame.reward}'
            fig = frame.q_cm_heatmap(title)
            fig.show()
    

    def CM_path(self):
        actions = ""
        states = ""
        cm_path = np.zeros((self.env_manager.obs_space.n,self.env_manager.obs_space.n))
        cm_path_annot = pd.DataFrame([['']*self.env_manager.obs_space.n]*self.env_manager.obs_space.n)

        for i in range(len(self.frames)-1):
            frame = self.frames[i]
            next_frame = self.frames[i+1]

            cm_path[frame.state, next_frame.state]+=1
            cm_path_annot.loc[frame.state, next_frame.state]+=str(i+1)
            actions += str(frame.action) if frame.action != None else ''
            states += f"{frame.state}, "
        actions += str(next_frame.action)
        states += str(next_frame.state)


        fig = Tools.heatmap_fig(z = cm_path, hover_text=cm_path, title=states)
        fig.show()
