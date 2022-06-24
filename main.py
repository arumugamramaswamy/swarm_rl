from stable_baselines3 import PPO
from policy import CustomMeanEmbeddingsExtractor

from to_vec_env import to_vec_env
from viz import test

import env as custom_simple_spread

env = custom_simple_spread.parallel_env(N=2, local_ratio=0, shuffle=True)
env = to_vec_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomMeanEmbeddingsExtractor,
    features_extractor_kwargs=dict(
        observation_space=env.observation_space,
        keys=list(env.observation_space.keys()),
        mean_keys={"entity_pos", "other_pos", "comm"}
    )
)

# ppo = PPO.load("models/model.zip")
ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=".")
ppo.learn(1_000_000, tb_log_name="tb_logs/simple_spread_dict")
ppo.save("models/model_2")

print(test(env, ppo, render=True))
