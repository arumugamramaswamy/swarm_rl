from experimenter.config_node import ConfigNode
from env import REGISTRY as ENV_REGISTRY
from policy import REGISTRY as FEATURE_EXTRACTOR_REGISTRY
from env.to_vec_env import to_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, A2C
from viz import test as test_

import os

EVAL_DIR = "eval"

ALGO_REGISTRY = {"ppo": PPO, "a2c": A2C}


def build_viz(cfg: ConfigNode, model_path):
    env = _prep_env(cfg.env)
    algo_constructor = ALGO_REGISTRY[cfg.algo]
    algo = algo_constructor.load(model_path, env)

    def viz():
        test_(env, algo, True)

    return viz


def parse_config(cfg: ConfigNode, experiment_dir):
    env = _prep_env(cfg.env)
    eval_env = _prep_env(cfg.eval_env)
    eval_cb = _prep_eval_cb(cfg.eval_cb_kwargs, eval_env, experiment_dir)
    algo_kwargs = cfg.get("algo_kwargs", {})
    algo_constructor = ALGO_REGISTRY[cfg.algo]
    if "checkpoint_path" in cfg:
        algo = algo_constructor.load(cfg.checkpoint_path, env)
        if "train_only_feature_extractor" in cfg and cfg.train_only_feature_extractor:
            for params in algo.policy.parameters():
                params.requires_grad = False

            for params in algo.policy.features_extractor.parameters():
                params.requires_grad = True
    else:
        policy_kwargs = _prep_policy_kwargs(cfg.policy_kwargs)
        algo = algo_constructor(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=experiment_dir,
            **algo_kwargs
        )

    def learn():
        algo.learn(cfg.num_timesteps, callback=eval_cb)
        algo.save(os.path.join(experiment_dir, "final_model"))

    def test():
        print(test_(env, algo, render=True))

    return learn, test


def _prep_env(cfg: ConfigNode):
    env_constructor = ENV_REGISTRY[cfg.env_name]
    return to_vec_env(env_constructor(**cfg.env_kwargs))


def _prep_eval_cb(cfg: ConfigNode, eval_env, experiment_dir):
    eval_path = os.path.join(experiment_dir, EVAL_DIR)
    extra_kwargs = ConfigNode(
        {
            "eval_env": eval_env,
            "best_model_save_path": eval_path,
            "log_path": eval_path,
        }
    )
    cfg.merge(extra_kwargs)
    return EvalCallback(**cfg)


def _prep_policy_kwargs(cfg: ConfigNode):
    feature_extractor_constructor = FEATURE_EXTRACTOR_REGISTRY[
        cfg.feature_extractor_name
    ]
    policy_kwargs = dict(cfg)
    special_kwargs = dict(
        features_extractor_class=feature_extractor_constructor,
        features_extractor_kwargs=cfg.feature_extractor_kwargs,
    )
    policy_kwargs.update(special_kwargs)

    # Weird bug related to naming in config
    policy_kwargs.pop("feature_extractor_name")
    policy_kwargs.pop("feature_extractor_kwargs")

    return policy_kwargs
