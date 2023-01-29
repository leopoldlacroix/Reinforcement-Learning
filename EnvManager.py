import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EnvManager:
    # instance = None

    # def get_instance():
    #     if EnvManager.instance != None:
    #         return EnvManager.instance
            

    def __init__(self, env_name):
        self.env : gym.Env = gym.make(env_name, render_mode="ansi", max_episode_steps = 200, is_slippery=False)
        self.P = self.env.P
        self.n_obs_space = self.env.observation_space.n
        self.n_action_space = self.env.action_space.n

        self.cm_rewards = np.zeros((self.env.observation_space.n, self.env.observation_space.n))
        self.action_matrix = pd.DataFrame([['']*self.env.observation_space.n]*self.env.observation_space.n)
       
        for state in range(self.action_matrix.shape[0]):
            for action in list(self.env.P[state].keys())[::-1]:
                for p, next_state, reward, done in self.env.P[state][action]:
                    self.action_matrix.loc[state, next_state] += str(action)
                    reward = -1 if done and reward < 1 else reward
                    self.cm_rewards[state, next_state] = reward
        self.action_matrix = self.action_matrix.replace({'': '-1'}).astype(int)
    
    def render(self):
        return self.env.render()
    
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def cm_rewards_heatmap(self):
        plt.figure()
        sns.heatmap(
            self.cm_rewards,
            annot = self.action_matrix
        )
        plt.show()

