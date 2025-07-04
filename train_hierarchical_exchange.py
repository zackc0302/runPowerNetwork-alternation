# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
import os
import ray
import logging
import wandb
import argparse
import yaml
import random
import numpy as np
import torch 
import pickle

import gym
from gym.spaces import Discrete, Tuple, Dict, Box

from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo, sac  # import the type of agents
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from ray import tune
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from dotenv import load_dotenv # security keys

from models.mlp import SimpleMlp, ChooseSubstationModel, ChooseActionModel
from models.substation_module import RllibSubsationModule
from models.hierarchical_agent import HierarchicalAgent, GreedySubModelNoWorker
from grid2op_env.grid_to_gym import Grid_Gym, Grid_Gym_Greedy, HierarchicalGridGym
from experiments.preprocess_config import preprocess_config, get_loader
from experiments.stopper import MaxNotImprovedStopper

from experiments.callback import CombinedCallbacks, LogDistributionsCallback, CustomSyncCallback

load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


logging.basicConfig(
    format='[INFO]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in \
    function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("choose_action"):
            return "choose_action_agent"
        else:
            return "choose_substation_agent"
            
LOCAL_DIR = "log_files"

if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)
    ModelCatalog.register_custom_model("fcn", SimpleMlp)
    ModelCatalog.register_custom_model("substation_module", RllibSubsationModule)
    ModelCatalog.register_custom_model("choose_substation_model", ChooseSubstationModel)
    ModelCatalog.register_custom_model("choose_action_model", ChooseActionModel)

    register_env("Grid_Gym", Grid_Gym)
    register_env("Grid_Gym_Greedy", Grid_Gym_Greedy)
    register_env("HierarchicalGridGym", HierarchicalGridGym)
    ray.shutdown()
    ray.init(ignore_reinit_error=False)

    parser = argparse.ArgumentParser(description="Train an agent on the Grid2Op environment")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm to use", choices=["ppo", "sac"])
    parser.add_argument("--algorithm_config_path", type=str, default="experiments/ppo/ppo_config.yaml", \
                                                         help="Path to config file for the algorithm")
    parser.add_argument("--use_tune", type=bool, default=True, help="Use Tune to train the agent")
    parser.add_argument("--project_name", type=str, default="testing_callback_grid", help="Name of the to be saved in WandB")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations to train the agent for.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers to use for training.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to use for training.")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Number of iterations between checkpoints.")
    parser.add_argument("--group" , type=str, default=None, help="Group to use for training.")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from a checkpoint. If yes, group must be specified.")
    parser.add_argument("--grace_period", type = int, default = 400, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--num_iters_no_improvement", type = int, default = 200, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--seed", type = int, default = -1, help = "Seed to use for training.")
    parser.add_argument("--with_opponent", type = bool, default= -1, help = "Whether to use an opponent or not.")
    parser.add_argument("--sub_freq", type=int, default=2, help="Every how many iterations to update substation agent (a in a:b)")
    parser.add_argument("--action_freq", type=int, default=1, help="Every how many iterations to update action agent (b in a:b)")
    parser.add_argument("--update_mode", type=str, default="proportional", choices=["proportional", "alternating"], help="Update strategy: proportional (default) or alternating.")

    args = parser.parse_args()

    logging.info("Training the agent with the following parameters:")

    for arg in vars(args):
        logging.info(f"{arg.upper()}: {getattr(args, arg)}")

    config = preprocess_config(yaml.load(open(args.algorithm_config_path), Loader=get_loader()))["tune_config"]
    config["callbacks"] = lambda: CombinedCallbacks(
        sub_update_every=args.sub_freq,
        action_update_every=args.action_freq,
        update_mode=args.update_mode  # "alternating" or "proportional"
    )

    if args.num_workers != -1:
        config["num_workers"] = args.num_workers

    if args.with_opponent != -1:
        config["env_config"]["with_opponent"] = True
        config["evaluation_config"]["env_config"]["with_opponent"] = True

    print("Config is", config)
    for key, val in config.items():
        print(f"Key {key} has value {val}")
        print("-" * 20)

    if args.algorithm == "ppo":
        trainer_cls = ppo.PPOTrainer
    elif args.algorithm == "sac":
        trainer_cls = sac.SACTrainer
    else:
        raise ValueError("Unknown algorithm. Choices are: ppo, sac")

    env_config_train = config["env_config"]
    env_config_val = config["evaluation_config"]["env_config"]

    grid_gym = Grid_Gym(env_config_train)

    substation_config = config["multiagent"]["policies"]["choose_substation_agent"]["config"]

    config["multiagent"]["policies"]["choose_substation_agent"] = (
        PPOTorchPolicy,
        gym.spaces.Dict({
            "regular_obs": grid_gym.observation_space,
            "chosen_action": Discrete(grid_gym.action_space.n),
        }),
        Discrete(8),
        {
            **substation_config,
            "train_batch_size": 512,
            "loss_fn": "experiments.custom_losses.custom_substation_loss"
        }
    )

    action_config = config["multiagent"]["policies"]["choose_action_agent"]["config"]

    config["multiagent"]["policies"]["choose_action_agent"] = (
        PPOTorchPolicy,
        gym.spaces.Dict({
            "action_mask": Box(0, 1, shape=(grid_gym.action_space.n,), dtype=np.float32),
            "regular_obs": grid_gym.observation_space,
            "chosen_substation": Discrete(8)
        }),
        grid_gym.action_space,
        {
            **action_config,
            "train_batch_size": 256,
            "loss_fn": "experiments.custom_losses.custom_action_loss"
        }
    )

    config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    if args.use_tune:
        reporter = CLIReporter()
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter=args.num_iters),
            MaxNotImprovedStopper(metric="episode_reward_mean",
                                grace_period=args.grace_period,
                                num_iters_no_improvement=args.num_iters_no_improvement,
                                no_stop_if_val=5500)
        )

        analysis = ray.tune.run(
            trainer_cls,
            progress_reporter=reporter,
            config=config,
            name=args.group,
            local_dir=LOCAL_DIR,
            checkpoint_freq=args.checkpoint_freq,
            stop=stopper,
            checkpoint_at_end=True,
            num_samples=args.num_samples,
            callbacks=[
                WandbLoggerCallback(
                    project=args.project_name,
                    group=args.group,
                    api_key=WANDB_API_KEY,
                    log_config=True
                )
            ],
            keep_checkpoints_num=5,
            checkpoint_score_attr="evaluation/episode_reward_mean",
            verbose=1,
            resume=args.resume
        )
        ray.shutdown()
    else:
        trainer_object = trainer_cls(env=Grid_Gym, config=config)

        for step in range(args.num_iters):
            result = trainer_object.train()
            print(result["episode_len_mean"], flush=True)
            if (step + 1) % args.checkpoint_freq == 0:
                checkpoint = trainer_object.save()
                print("checkpoint saved at", checkpoint)
            print("-" * 40, flush=True)