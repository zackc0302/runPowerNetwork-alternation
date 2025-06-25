# Hierarchical RL with Alternating Updates for Power Network Topology

This repository builds upon and modifies the code from [bmanzack's runPowerNetworks](https://github.com/bmanczak/runPowerNetworks)

Aimming to use **alternation update** and **block coordinate gradient descent** to stablize the powerNetwork

## Paper

**PENDING**

### Create the environment (contributed by yensh) on PC

> The version of each package is very important. ***We are still working on updating all of the packages to the latest versions***.
> 
```
conda create -n rlib_grid python=3.7.9 -y
conda activate rlib_grid

conda install pytorch==1.10.0 -c pytorch

pip install Grid2Op==1.6.4 lightsim2grid==0.5.4

pip install ray==1.9.0

pip install protobuf==3.20.0
pip install importlib-metadata==4.13.0

pip install gym==0.21.0 tabulate==0.8.9 numba==0.54.1 tqdm==4.62.3 
pip install pillow==8.0.1 dm_tree scikit-image==0.18.3 lz4==3.1.3
pip install python-dotenv tensorboardX==2.4
pip install wandb
```

### Set the Grid2Op environment

You can run the `setup_grid2op.py` script to set up the environment.

### Modify the models/utils.py
```
# from sknetwork.utils import edgelist2adjacency
def edgelist2adjacency(edge_list, num_nodes=None):
    """
    Convert an edge list to a sparse adjacency matrix.

    Parameters:
    - edge_list: list of (i, j) pairs
    - num_nodes: total number of nodes (optional)

    Returns:
    - scipy.sparse.coo_matrix adjacency matrix
    """
    if num_nodes is None:
        num_nodes = max(max(i, j) for i, j in edge_list) + 1

    row = [i for i, j in edge_list]
    col = [j for i, j in edge_list]
    data = np.ones(len(edge_list))

    adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj
```


### Agent training

Note that wandb is used for monitoring the progress of the experiment.
If you wish to use wandb make sure to specify the `WANDB_API_KEY` in the `.env` file. Alternatively, comment out `WandbLoggerCallback` in the `train.py` file.

#### Setup

We train and benchmark the models in an environment with and without outages. The environment setting is controlled by the boolean `--with_opponennt` keyword argument in the `train.py` script.

By default, the 5 best checkpoints in terms of mean episode reward will be saved in the `log_files` directory.

#### Native and hybrid agents

These agents support training with *PPO* and *SAC* algorithms. 
To train these agents, go to the `main` branch and run the `train.py` file with desired keyword arguments. The choice of hyperparameters in a `.yaml` file. Specifications used in the paper are found in the `experiments` folder under the corresponding algorithm name.

For instance, to train a hybrid PPO agent in the setting with outages for 1000 iterations and over 10 different seeds run:

``` 
python train.py --algorithm ppo \
 --algorithm_config_path experiments/ppo/ppo_run_threshold.yaml \
 --with_opponent True \
 --num_iters 1000 \
 --num_samples 10 \
 ```

See the `argparse` help for more details on keyword arguments.

#### Hierarchical agent

To train a fully hierarchical agent go to the `hierarchical_approach` branch and run the `train_hierarchical.py` file with desired keyword arguments. Similar to the native and hybrid agents, the choice of hyperparameters in a `.yaml` file.

To train the hierarchical agent in the setting with outages for 1000 iterations and over 10 different seeds run:

```
python train_hierarchical_exchange.py --algorithm ppo \
 --algorithm_config_path experiments/hierarchical/full_mlp_share_critic.yaml \
 --use_tune True \
 --num_iters 1000 \
 --num_samples 16 \
 --with_opponent True \
 --project_name 
```

- 只用 a:b 頻率控制（非互斥）：
```
python train_hierarchical.py --sub_freq 3 --action_freq 1
```
- 啟用交錯互斥更新（中、下輪流更新）：
```
python train_hierarchical_exchange.py \
  --algorithm ppo \
  --algorithm_config_path experiments/hierarchical/full_mlp_share_critic.yaml \
  --use_tune True \
  --num_iters 1000 \
  --num_samples 3 \
  --with_opponent True \
  --project_name alter_hierarchical_mutual_exclusive \
  --sub_freq 1 \
  --action_freq 1 \
  --mutual_exclusive
```

### Evaluation

To run the trained agent on the set of test chronics run:

```
python evaluation/run_eval.py \
  --agent_type ppo \
  --checkpoint_path log_files/PPO_2025-05-13_15-03-37/PPO_HierarchicalGridGym_646ec_00000_0_seed=0_2025-05-13_15-03-37 \
  --checkpoint_num 270 \
  --use_split val \
  --hierarchical True
```
If the agent being evaluated is a fully hierarchical (i.e. non-hybrid) add keyword argument `--hierarchical True`.
Except for printing the mean episode length, this script involves data collection that is needed for further analysis. The data is saved in a folder `evaluation/eval_results` and can be used for further analysis.

The functionality for further analysis is implemented in the `evaluation/results_analysis.py` file. Given the path to evaluation results it is easy to obtain a table with the statistics:

```
from evaluation.result_analysis import process_eval_data_multiple_agents, \
 get_analysis_objects, \
 compile_table_df

EVAL_PATHS = {"Agent Type 1": (path_to_eval_results, "wandb_num"),
 "Agent Type 2": (path_to_eval_results, "wandb_num"), ...}

data_per_algorithm = process_eval_data_multiple_agents(EVAL_PATHS)

# Compile the data frame from which we will later plot the results
df = compile_table_df(data_per_algorithm)
```
### Repository overview 

`evaluation` contains the code for benchmarking trained agents

`experiments` contains the specification of the model hyperparameters and custom callbacks 

`grid2op_env` contains the environment wrappers, train/test/val split, and data used to scale the observations

`models` contains the code for the torch models used in the experiments

`notebooks` contains miscellaneous notebooks used in the course of development and evaluation. Notably `sub_node_model.ipnyb` contains an alpha version of a Graph Neural Network (GNN) based policy.
 
@misc{manczak2023hierarchical,
      title={Hierarchical Reinforcement Learning for Power Network Topology Control}, 
      author={Blazej Manczak and Jan Viebahn and Herke van Hoof},
      year={2023},
      eprint={2311.02129},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
