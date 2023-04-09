from nht.interface_wrappers.nht_wrapper import register_NHT_env
import gym

register_NHT_env('HalfCheetah-v4', 'NHT/test_scripts/map_model/NHT-L_10')

my_NHT_env = gym.make('NHT_HalfCheetah-v4')