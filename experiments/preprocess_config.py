from numpy import save
import yaml
from grid2op_env.grid_to_gym import Grid_Gym, HierarchicalGridGym
from experiments.callback import CombinedCallbacks, LogDistributionsCallback, CustomSyncCallback

from ray import tune
from experiments.callback import CustomSyncCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from experiments.callback import CombinedCallbacks

mapper = {
    "callbacks": {
        "CombinedCallbacks": CombinedCallbacks,
        "LogDistributionsCallback": LogDistributionsCallback,
        "experiments.callback.CombinedCallbacks": CombinedCallbacks,
        "experiments.callback.LogDistributionsCallback": LogDistributionsCallback
    },
    "env":{
        "Grid_Gym": Grid_Gym,
        "HierarchicalGridGym": HierarchicalGridGym,
    }
}
def preprocess_config(config):
    """
    Transform the string representations of classes in YAML
    files to the corresponding python objects.

    Args:
        config (dict): parsed YAML config file
    """
    if "callbacks" in config["tune_config"]:
        if isinstance(config["tune_config"]["callbacks"], list):
            callbacks_lst = []
            for callback_name in config["tune_config"]["callbacks"]:
                callbacks_lst.append(mapper["callbacks"][callback_name])  
            config["tune_config"]["callbacks"] = callbacks_lst
        else:  # single str 
            config["tune_config"]["callbacks"] = mapper["callbacks"][config["tune_config"]["callbacks"]]
        
    try:
        config["tune_config"]["env"] = mapper["env"][config["tune_config"]["env"]]
    except:
        pass

    # 修正 multi-agent policies
    if "multiagent" in config["tune_config"]:
        env_instance = config["tune_config"]["env"](config["tune_config"]["env_config"])
        obs_space = env_instance.observation_space
        action_space = env_instance.action_space

        policies = config["tune_config"]["multiagent"].get("policies", {})
        new_policies = {}
        for policy_id, policy_cfg in policies.items():
            if isinstance(policy_cfg, dict):
                new_policies[policy_id] = (None, obs_space, action_space, policy_cfg["config"])
            else:
                new_policies[policy_id] = policy_cfg  # 保留原格式
        config["tune_config"]["multiagent"]["policies"] = new_policies

    return config


    
def tune_search_quniform_constructor(loader, node):
    """
    Constructor for tune uniform float sampling  

    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.quniform(vals[0], vals[1], vals[2])

def tune_search_grid_search_constructor(loader, node):
    """
    Constructor for tune grid search.

    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.grid_search(vals)

def tune_choice_constructor(loader, node):
    """
    Constructor for tune grid search.

    """
    vals = []
    for scalar_node in node.value:
        if scalar_node.value == "True":
            val = True
        elif scalar_node.value == "False":
            val = False
        else: 
            val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.choice(vals)
    
def get_loader():
  """Add constructors to PyYAML loader."""
  loader = yaml.SafeLoader
  loader.add_constructor("!quniform", tune_search_quniform_constructor)
  loader.add_constructor("!grid_search", tune_search_grid_search_constructor)
  loader.add_constructor("!choice", tune_choice_constructor)
  return loader

def float_to_integer(float_value):
    if float_value.is_integer():
        return int(float_value)
    else:
        return float_value
