import gym
import numpy as np
import pandas as pd

import plotly.express as px
from Tools import *

from gym.spaces import Space

class EnvManager:
    # instance = None

    # def get_instance():
    #     if EnvManager.instance != None:
    #         return EnvManager.instance
            

    def __init__(self, env_name, **kargs):
        self.env : gym.Env = gym.make(env_name, render_mode="ansi", **kargs)
        self.P = self.env.P
        self.obs_space: Space = self.env.observation_space
        self.action_space: Space = self.env.action_space

        """Adjency matrix of rewards"""
        self.cm_rewards = np.zeros((self.obs_space.n, self.obs_space.n))
       
        """Text to display on top of adjency matrix"""
        self.hover_matrix = pd.DataFrame([f'current_state: {i} \n next_state: {j} \n actions: ' for i in range(self.obs_space.n)] for j in range(self.obs_space.n))
       
        for state in range(self.hover_matrix.shape[0]):
            for action in list(self.env.P[state].keys())[::-1]:
                for p, next_state, reward, done in self.env.P[state][action]:
                    self.hover_matrix.loc[state, next_state] += f'{action}'
                    reward = -1 if done and reward < 1 else reward
                    self.cm_rewards[state, next_state] = reward
        self.hover_matrix = self.hover_matrix
    
    def render(self):
        return self.env.render()
    
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)


    def cm_rewards_heatmap(self) -> Figure:
        return Tools.heatmap_fig(
            z = self.cm_rewards, 
            hover_text=self.hover_matrix,
            title = "Reward adjency matrix"
        )
