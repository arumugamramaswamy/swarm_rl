from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_latest_run_id

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from policy.policy import FullSelfAttentionSimpleSpread, CrossAttentionSimpleSpread
from policy.attention_policy import AttentionPolicyV1

from env.to_vec_env import to_vec_env
from viz import test

import env.env as custom_simple_spread
import os

TB_LOG_DIR = "tb_logs"
EVAL_DIR = "eval"

EXP_NAME = "attention_embeddings"
N = 3

run_id = get_latest_run_id(TB_LOG_DIR, EXP_NAME) + 1
run_name = f"{EXP_NAME}_{run_id}"
eval_path = os.path.join(EVAL_DIR, run_name)

env = custom_simple_spread.parallel_env(N=N, local_ratio=0, shuffle=True)
env = to_vec_env(env)

policy_kwargs = dict(
    # features_extractor_class=CustomAttentionMeanEmbeddingsExtractorSimpleSpread,
    features_extractor_class=AttentionPolicyV1,
    features_extractor_kwargs=dict(
        # keys=list(env.observation_space.keys()),
        # embedding_size=32
    ),
)

eval_env = custom_simple_spread.parallel_env(N=N, local_ratio=0, shuffle=True)
eval_env = to_vec_env(eval_env)

eval_callback_kwargs = {
    "eval_env": eval_env,
    "best_model_save_path": eval_path,
    "log_path": eval_path,
    "deterministic": True,
    "eval_freq": 10000,
    "n_eval_episodes": 100,
    "render": False,
}

eval_cb = EvalCallback(**eval_callback_kwargs)


def train_ppo():
    # ppo = PPO.load("models/model_2_me.zip")
    # ppo.env = env
    ppo = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )
    ppo.learn(6_000_000, tb_log_name=EXP_NAME, callback=eval_cb)
    ppo.save(os.path.join("models", run_name))

    print(test(env, ppo, render=True))


def train_recurrent_ppo():
    ppo = RecurrentPPO(
        MultiInputLstmPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )
    ppo.learn(6_000_000, tb_log_name=EXP_NAME, callback=eval_cb)
    ppo.save(os.path.join("models", run_name))

    print(test(env, ppo, render=True))


train_ppo()
