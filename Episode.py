from Agent import Agent
from IPython.display import clear_output

from EnvManager import *


class Frame:
    def __init__(self, env_manager: EnvManager, state, action, reward, q_table, done, truncated):
        self.env_manager = env_manager
        self.rep = env_manager.render()
        self.state = state
        self.action = action
        self.reward = reward
        self.q_table = q_table
        self.done = done
        self.truncated = truncated
        self.q_CM = self.weigthed_connection_matrix_from_q_table()
    
    def weigthed_connection_matrix_from_q_table(self):
        lc = self.env_manager.n_obs_space
        connection_matrix: np.ndarray = np.zeros((lc, lc))
        for state in self.env_manager.P:
            for action in list(self.env_manager.P[state].keys())[::-1]:
                for p, next_state, reward, done in self.env_manager.P[state][action]:
                    connection_matrix[state, next_state] = self.q_table[state, action]

        return connection_matrix
    
    def q_cm_heatmap(self):
        return sns.heatmap( 
            self.q_CM,
            annot = self.env_manager.action_matrix,
            fmt='.5g',
        )

    def __repr__(self) -> str:
        return f"state: {self.state}, action: {self.action}, reward: {self.reward}, truncated: {self.truncated}, done: {self.done}"
    

class Episode:
    def __init__(self, env_manager: EnvManager, agent: Agent):
        self.frames:list[Frame] = []
        self.env_manager = env_manager
        state, info = self.env_manager.reset()
        agent.set_state(state)
        
        done, truncated, reward, action = False, False, None, None
        self.add_frame(self.env_manager, state, action, reward, agent._q_table, done, truncated)
        while not (done or truncated):
            action = agent.action()
            state, reward, done, truncated, info = self.env_manager.step(action)
            reward = -1 if done and reward < 1 else reward
            agent.update(reward, state)
            self.add_frame(self.env_manager, state, action, reward, agent._q_table, done, truncated)
        
            
        agent.update(reward, state)
        self.done = done
        self.truncated = truncated

    
    def add_frame(self, env_manager, state, action, reward, q_table, done, truncated):
        self.frames.append(
            Frame(
            env_manager= env_manager,
                state = state,
                action = action,
                reward = reward,
                q_table = q_table,
                done = done, 
                truncated = truncated
            )
        )
    def print_frames(self):
        for i, frame in enumerate(self.frames):
            print(f"Reward: {frame.reward}")
            print(f"State: {frame.state}")
            print(f"states q_table : {frame.q_table[frame.state]}")
            print(f"Action: {frame.action}")
            print(frame.rep)
            print("------------------")

    def progress_animation(self, all = False):
        for frame in self.frames:
            if not all:
                clear_output(wait=True)
            plt.figure()
            frame.q_cm_heatmap()
            plt.title(f'state: {frame.state}, action: {frame.action}, reward: {frame.reward}')
            plt.show()
    
    def CM_path(self):
        actions = ""
        cm_path = np.zeros((self.env_manager.n_obs_space,self.env_manager.n_obs_space))
        cm_path_annot = pd.DataFrame([['0']*self.env_manager.n_obs_space]*self.env_manager.n_obs_space)
        cm_path[0,0]+=1
        cm_path_annot.iloc[0,0] += '1'
        for i in range(len(self.frames)-1):
            frame = self.frames[i]
            next_frame = self.frames[i+1]

            cm_path[frame.state, next_frame.state]+=1
            cm_path_annot.loc[frame.state, next_frame.state]+=str(i+2)
            actions += str(frame.action) if frame.action != None else ''
        actions += str(next_frame.action)

        sns.heatmap( 
            cm_path,
            annot=cm_path_annot.astype(int),
            fmt='.5g',
        )
        plt.title(actions)
        plt.show()
