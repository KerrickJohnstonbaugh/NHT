from gym import utils as gym_utils
from gym import spaces
import numpy as np
from gym.envs.registration import register
import gym
from gym.envs.registration import spec, load
import tensorflow as tf
from baselines.common.tf_util import get_session
from nht.LASER import LASER

def register_LASER_env(base_env, LASER_path, action_dim):
    
    temp_env = gym.make(base_env)

    register(
        id=f'LASER_{base_env}',
        entry_point='nht.interface_wrappers.laser_wrapper:LASERwrapper',
        kwargs={'env': temp_env, 'LASER_path': LASER_path, 'action_dim': action_dim},
    )


class LASERwrapper(gym.Wrapper):
    def __init__(self, env, LASER_path, action_dim):
        super().__init__(env)
        
        self.action_dim = action_dim
        cond_size = self.env.observation_space.shape[0]
        u_dim = self.env.action_space.shape[0]

        self.tfsess = get_session()
        # placeholders to construct NHT computational graph
        self.cond_inp = tf.placeholder(shape=[None, cond_size], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)

        # LASER model
        self.LASER_model = LASER(input_dim=u_dim, latent_dim=action_dim, cond_dim=cond_size)

        self.LASER_model.decoder.net = tf.keras.models.load_model(LASER_path)
        self.decoded_action = tf.stop_gradient(self.LASER_model.decoder(tf.concat((self.z, self.cond_inp),axis=-1)))
        self.LASER_model.freeze_model()
        self.set_action_space()


    def set_action_space(self):
        n = self.action_space.shape[0]
        self.action_space = spaces.Box(low=-np.sqrt(n/self.action_dim), high=np.sqrt(n/self.action_dim), shape=(self.action_dim,), dtype=np.float32)

    def step(self, action):

        k = self.action_dim
        assert action.shape == (k,)

        o_r = np.expand_dims(self._get_obs().copy(),0)
        action = self.tfsess.run(self.decoded_action, feed_dict={self.cond_inp: o_r, self.z: np.expand_dims(action.copy(),0)})
        action = action.squeeze().copy()

        return self.env.step(action)