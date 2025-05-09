import torch
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss

def custom_substation_loss(policy, model, dist_class, train_batch):
    if not policy.config.get("substation_update_allowed", True):
        # 跳過 loss → 不讓 optimizer 做事
        return torch.tensor(0.0, requires_grad=True)
    else:
        return ppo_surrogate_loss(policy, model, dist_class, train_batch)
