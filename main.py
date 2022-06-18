from stable_baselines3 import PPO

from to_vec_env import to_vec_env

import env as custom_simple_spread

env = custom_simple_spread.parallel_env()
env = to_vec_env(env)

ppo = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=".")

ppo.learn(1_000_000, tb_log_name="simple_spread_dict")

ppo.save(".")
