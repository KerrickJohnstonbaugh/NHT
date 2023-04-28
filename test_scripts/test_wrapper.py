from nht.interface_wrappers.nht_wrapper import register_NHT_env
import gym

register_NHT_env('Walker2d-v2', 'NHT/test_scripts/map_model/NHT-L_10', action_dim=2)

my_NHT_env = gym.make('NHT_Walker2d-v2')
print(my_NHT_env.reset())
print(my_NHT_env.step([0, 0]))