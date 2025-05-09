# experiments/custom_ppo_trainer.py
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from grid2op_env.grid_to_gym import Grid_Gym

class CustomPPOTrainer:
    def __init__(self, env_config):
        config = PPOConfig().environment(
            env=Grid_Gym,
            env_config=env_config
        )
        self.algo = config.build()

    def train(self):
        return self.algo.train()

    def save(self, checkpoint_dir=None):
        return self.algo.save(checkpoint_dir)

    def restore(self, checkpoint_path):
        self.algo.restore(checkpoint_path)
