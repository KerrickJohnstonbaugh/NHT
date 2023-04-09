from gym import utils as gym_utils
from gym import spaces
import numpy as np
from gym.envs.registration import register
import gym
from gym.envs.registration import spec, load
from nht.utils import load_interface
import torch

def register_NHT_env(base_env, NHT_path):
    
    temp_env = gym.make(base_env)

    register(
        id=f'NHT_{base_env}',
        entry_point='nht.mujoco_interface:NHTwrapper',
        kwargs={'env': temp_env, 'NHT_path': NHT_path},
    )


class NHTwrapper(gym.Wrapper):
    def __init__(self, env, NHT_path):
        super().__init__(env)
        
        model_path = NHT_path
        model_type = 'NHT'
        self.Q = load_interface(model_type, model_path)
        self.action_dim = self.Q.k
        self.set_action_space()


    def set_action_space(self):
        n = self.action_space.shape[0]
        self.action_space = spaces.Box(low=-np.sqrt(n/self.action_dim), high=np.sqrt(n/self.action_dim), shape=(self.action_dim,), dtype=np.float32)

    def step(self, action):

        k = self.action_dim
        assert action.shape == (k,)

        with torch.no_grad():
            c = torch.tensor(self.unwrapped._get_obs().copy(),dtype=torch.float32).unsqueeze(0)
            Q_hat = self.Q(c).squeeze(0)
            a = np.expand_dims(action.copy(),1) # turn action from agent to column vector tensor (with batch dimension)
            u = np.matmul(Q_hat, a).squeeze()

            action = u.numpy().copy()
            action = np.clip(action, -1, 1)

        return_values = self.env.step(action)

        return return_values