from gym import spaces
import numpy as np
from gym.envs.registration import register
import gym
import json

def register_SVD_env(base_env, SVD_path, action_dim):
    
    temp_env = gym.make(base_env)

    register(
        id=f'SVD_{base_env}',
        entry_point='nht.interface_wrappers.svd_wrapper:SVDwrapper',
        kwargs={'env': temp_env, 'SVD_path': SVD_path, 'action_dim': action_dim},
    )


class SVDwrapper(gym.Wrapper):
    def __init__(self, env, SVD_path, action_dim):
        super().__init__(env)
        
        self.action_dim = action_dim

        with open(SVD_path, 'r') as f:
            SVD_dict = json.load(f)

        self.U = np.array(SVD_dict['U'])
        self.set_action_space()


    def set_action_space(self):
        n = self.action_space.shape[0]
        self.action_space = spaces.Box(low=-np.sqrt(n/self.action_dim), high=np.sqrt(n/self.action_dim), shape=(self.action_dim,), dtype=np.float32)

    def step(self, action):

        k = self.action_dim
        assert action.shape == (k,)

        c = np.expand_dims(self._get_obs().copy(),0)
        Q_hat = self.tfsess.run(self.Q_hat, feed_dict={self.cond_inp: c})
        a = np.expand_dims(action.copy(),1) # turn action from agent to column vector tensor (with batch dimension)
        u = np.matmul(Q_hat.squeeze(0), a).squeeze()

        action = u.copy()

        return self.env.step(action)