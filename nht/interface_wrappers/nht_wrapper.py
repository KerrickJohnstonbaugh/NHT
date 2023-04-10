from gym import utils as gym_utils
from gym import spaces
import numpy as np
from gym.envs.registration import register
import gym
from gym.envs.registration import spec, load
import tensorflow as tf
from baselines.common.tf_util import get_session
from nht.NHT import NHT

def register_NHT_env(base_env, NHT_path, action_dim):
    
    temp_env = gym.make(base_env)

    register(
        id=f'NHT_{base_env}',
        entry_point='nht.interface_wrappers.nht_wrapper:NHTwrapper',
        kwargs={'env': temp_env, 'NHT_path': NHT_path, 'action_dim': action_dim},
    )


class NHTwrapper(gym.Wrapper):
    def __init__(self, env, NHT_path, action_dim):
        super().__init__(env)
        
        self.action_dim = action_dim
        cond_size = self.env.observation_space.shape[0]
        u_dim = self.env.action_space.shape[0]

        self.tfsess = get_session()
        # placeholders to construct NHT computational graph
        self.cond_inp = tf.placeholder(shape=[None, cond_size], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)

        # NHT model (outputs basis for actuation subspace given current observation)
        self.NHT_model = NHT(action_dim=action_dim, output_dim=u_dim, cond_dim=cond_size)

        
        self.NHT_model.h.net = tf.keras.models.load_model(NHT_path) # loads weights
        self.Q_hat = tf.stop_gradient(self.NHT_model._get_map(self.cond_inp))
        self.NHT_model.freeze_model()
        self.set_action_space()


    def set_action_space(self):
        n = self.action_space.shape[0]
        self.action_space = spaces.Box(low=-np.sqrt(n/self.action_dim), high=np.sqrt(n/self.action_dim), shape=(self.action_dim,), dtype=np.float32)

    def step(self, action):

        k = self.action_dim
        assert action.shape == (k,)

        c = np.expand_dims(self.unwrapped._get_obs().copy(),0)
        Q_hat = self.tfsess.run(self.Q_hat, feed_dict={self.cond_inp: c})
        a = np.expand_dims(action.copy(),1) # turn action from agent to column vector tensor (with batch dimension)
        u = np.matmul(Q_hat.squeeze(0), a).squeeze()

        action = u.copy()

        return self.env.step(action)