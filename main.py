from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from policy import CustomAttentionMeanEmbeddingsExtractor

from to_vec_env import to_vec_env
from viz import test

import env as custom_simple_spread

N = 3
env = custom_simple_spread.parallel_env(N=N, local_ratio=0, shuffle=True)
env = to_vec_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomAttentionMeanEmbeddingsExtractor,
    features_extractor_kwargs=dict(
        keys=list(env.observation_space.keys()),
        mean_keys={"entity_pos", "other_pos", "comm"}
    )
)

eval_env = custom_simple_spread.parallel_env(N=N, local_ratio=0, shuffle=True)
eval_env = to_vec_env(eval_env)

eval_callback_kwargs = {
    "eval_env": eval_env,
    "best_model_save_path": "eval",
    "log_path": "eval",
    "deterministic": True,
    "eval_freq": 10000,
    "n_eval_episodes": 100,
    "render": False
}

eval_cb = EvalCallback(**eval_callback_kwargs)

# ppo = PPO.load("models/model_2_me.zip")
# ppo.env = env
ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="tb_logs")
ppo.learn(6_000_000, tb_log_name="attention_embeddings", callback=eval_cb)
ppo.save("models/attn_2")

print(test(env, ppo, render=True))
