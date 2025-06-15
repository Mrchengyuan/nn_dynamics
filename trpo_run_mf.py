import numpy as np # 导入numpy库 (Import numpy library)
# import matplotlib.pyplot as plt # 此脚本中未使用matplotlib (matplotlib is not used in this script)
import math # 导入math模块 (Import math module)
npr = np.random # 将numpy.random赋给npr (Assign numpy.random to npr)
import torch # 导入torch库 (Import torch library)
import torch.nn as nn # 导入torch.nn模块 (Import torch.nn module)
import torch.nn.functional as F # 导入torch.nn.functional模块 (Import torch.nn.functional module)
# from six.moves import cPickle # 此脚本中未使用cPickle (cPickle is not used in this script)
# from collect_samples import CollectSamples # 此脚本中未使用 (Not used in this script)
# from get_true_action import GetTrueAction # 此脚本中未使用 (Not used in this script)
import os # 导入os模块 (Import os module)
# import copy # 此脚本中未使用copy (copy is not used in this script)
from helper_funcs import create_env # 从helper_funcs导入create_env (Import create_env from helper_funcs)
import argparse # 导入argparse模块 (Import argparse module)

# Garage相关导入 (Garage imports)
from garage import wrap_experiment # 导入garage的wrap_experiment (Import wrap_experiment from garage)
from garage.envs import GymnasiumEnv # 导入garage的GymnasiumEnv包装器 (Import GymnasiumEnv wrapper from garage.envs)
from garage.experiment import Snapshotter # 导入garage的Snapshotter (Import Snapshotter from garage.experiment)
from garage.experiment.deterministic import set_seed # 导入garage的set_seed (Import set_seed from garage.experiment.deterministic)
from garage.torch.algos import TRPO as GarageTRPO # 导入garage的TRPO算法 (Import TRPO algorithm from garage)
from garage.torch.policies import GaussianMLPPolicy as GarageGaussianMLPPolicy # 导入garage的高斯MLP策略 (Import GaussianMLPPolicy from garage)
from garage.torch.value_functions import GaussianMLPValueFunction # 导入garage的高斯MLP价值函数 (Import GaussianMLPValueFunction from garage)
from garage.trainer import Trainer # 导入garage的Trainer (Import Trainer from garage)


@wrap_experiment # 使用garage的wrap_experiment装饰器 (Decorate with garage's wrap_experiment)
def run_task(ctxt, v_dict_args): # 定义运行任务的函数，ctxt由wrap_experiment提供 (Define function to run task, ctxt is provided by wrap_experiment)

    set_seed(v_dict_args['seed']) # 使用garage的set_seed设置随机种子 (Set random seed using garage's set_seed)
    env, _ = create_env(v_dict_args["which_agent"], seed=v_dict_args.get("seed", npr.randint(0,10000))) # 创建环境，传递种子 (Create environment, pass seed)
    env = GymnasiumEnv(env) # 使用GymnasiumEnv包装环境 (Wrap environment with GymnasiumEnv)

    policy = GarageGaussianMLPPolicy(env.spec, # 创建garage的高斯MLP策略 (Create garage's GaussianMLPPolicy)
                                     hidden_sizes=[64, 64], # 隐藏层大小 (Hidden layer sizes)
                                     hidden_nonlinearity=F.tanh, # 隐藏层激活函数 (Hidden layer activation function)
                                     output_nonlinearity=None) # 输出层激活函数 (Output layer activation function)

    value_function = GaussianMLPValueFunction(env_spec=env.spec, # 创建garage的高斯MLP价值函数 (Create garage's GaussianMLPValueFunction)
                                          hidden_sizes=(64, 64), # 隐藏层大小 (Hidden layer sizes)
                                          hidden_nonlinearity=F.tanh, # 隐藏层激活函数 (Hidden layer activation function)
                                          output_nonlinearity=None) # 输出层激活函数 (Output layer activation function)

    # 注意：Garage的TRPO可能需要不同的优化器设置或使用其内部默认值 (Note: Garage's TRPO might require different optimizer settings or use its internal defaults)
    # rllab的 ConjugateGradientOptimizer 和 FiniteDifferenceHvp 在garage中有其等价物或不同的实现方式
    # (rllab's ConjugateGradientOptimizer and FiniteDifferenceHvp have equivalents or different implementations in garage)
    # Garage TRPO的默认优化器通常是合适的 (Garage TRPO's default optimizers are usually suitable)

    trainer = Trainer(ctxt) # 创建garage的Trainer (Create garage's Trainer)

    algo = GarageTRPO(env_spec=env.spec, # 创建garage的TRPO算法实例 (Create garage's TRPO algorithm instance)
                      policy=policy, # 策略 (Policy)
                      value_function=value_function, # 价值函数 (Value function)
                      discount=0.995, # 折扣因子 (Discount factor)
                      # Garage TRPO 使用 policy_lr 和 vf_lr, 以及 optimizer_args 来配置优化器
                      # (Garage TRPO uses policy_lr and vf_lr, and optimizer_args to configure optimizers)
                      # step_size (在rllab中) 类似于Garage TRPO中的 policy_lr (step_size (in rllab) is similar to policy_lr in Garage TRPO)
                      # 对于更精细的控制，可以传递 optimizer 和 optimizer_args
                      # (For finer control, optimizer and optimizer_args can be passed)
                      )

    trainer.setup(algo, env) # 设置训练器 (Setup the trainer)
    trainer.train(n_epochs=v_dict_args["num_trpo_iters"], # 训练轮数 (Number of training epochs)
                  batch_size=v_dict_args["batch_size"], # 批处理大小 (Batch size)
                  # plot=True 参数在garage trainer.train中不可用，绘图通过TensorBoard或CSV完成
                  # (plot=True argument is not available in garage trainer.train, plotting is done via TensorBoard or CSVs)
                  )

##########################################
##########################################

# 指定参数 (ARGUMENTS TO SPECIFY)
parser = argparse.ArgumentParser() # 创建ArgumentParser对象 (Create ArgumentParser object)
parser.add_argument('--seed', type=int, default='0') # 添加seed参数 (Add seed argument)
parser.add_argument('--steps_per_rollout', type=int, default='1000') # 添加steps_per_rollout参数 (Add steps_per_rollout argument)
parser.add_argument('--save_trpo_run_num', type=int, default='1') # 添加save_trpo_run_num参数 (Add save_trpo_run_num argument)
parser.add_argument('--which_agent', type=int, default= 2) # 添加which_agent参数 (Add which_agent argument)
# parser.add_argument('--num_workers_trpo', type=int, default=2) # num_workers_trpo 在garage中通过采样器配置 (num_workers_trpo is configured via sampler in garage)
args = parser.parse_args() # 解析命令行参数 (Parse command-line arguments)

batch_size = 50000 # 批处理大小 (Batch size) # Garage TRPO中，这通常指每个epoch的样本总数 (In Garage TRPO, this usually refers to total samples per epoch)
# Garage的TRPO使用 rollouts_per_task * max_episode_length 来确定每个epoch的样本 (Garage's TRPO uses rollouts_per_task * max_episode_length to determine samples per epoch)
# batch_size 将传递给 trainer.train (batch_size will be passed to trainer.train)

steps_per_rollout = args.steps_per_rollout # 每回合步数 (Steps per rollout)
num_trpo_iters = 2500 # TRPO迭代次数 (Number of TRPO iterations)
if(args.which_agent==1): # 如果智能体是Ant (If agent is Ant)
	num_trpo_iters = 2500 # 设置TRPO迭代次数 (Set number of TRPO iterations)
if(args.which_agent==2): # 如果智能体是Swimmer (If agent is Swimmer)
	steps_per_rollout=333 # 设置每回合步数 (Set steps per rollout) # 这将是Garage中的max_episode_length (This will be max_episode_length in Garage)
	num_trpo_iters = 500 # 设置TRPO迭代次数 (Set number of TRPO iterations)
if(args.which_agent==4): # 如果智能体是HalfCheetah (If agent is HalfCheetah)
	num_trpo_iters= 2500 # 设置TRPO迭代次数 (Set number of TRPO iterations)
if(args.which_agent==6): # 如果智能体是Hopper (If agent is Hopper)
	num_trpo_iters= 2000 # 设置TRPO迭代次数 (Set number of TRPO iterations)

##########################################
##########################################

# 设置随机种子 (set random seed) - 现在由garage.experiment.deterministic.set_seed处理 (Now handled by garage.experiment.deterministic.set_seed)
# npr.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(args.seed)

# 构建variant字典 (Construct variant dictionary)
variant_dict=dict(batch_size=batch_size, which_agent=args.which_agent, # 变体参数字典 (Variant parameters dictionary)
            steps_per_rollout=steps_per_rollout, num_trpo_iters=num_trpo_iters, # 更多变体参数 (More variant parameters)
            # FiniteDifferenceHvp 和 ConjugateGradientOptimizer 是rllab特有的，garage TRPO有自己的实现 (FiniteDifferenceHvp and ConjugateGradientOptimizer are rllab specific, garage TRPO has its own)
            seed=args.seed) # 将种子传递给variant (Pass seed to variant)

# 构建实验名称 (Construct experiment name)
exp_name_str = f'agent_{args.which_agent}_seed_{args.seed}_mf_run{args.save_trpo_run_num}' # 实验名称字符串 (Experiment name string)

# 使用wrap_experiment运行任务 (Run task using wrap_experiment)
# snapshot_dir和exp_name将由wrap_experiment处理 (snapshot_dir and exp_name will be handled by wrap_experiment)
run_task(snapshot_config=Snapshotter(), #快照配置 (Snapshot configuration)
         variant=variant_dict, # 变体参数 (Variant parameters)
         exp_name=exp_name_str, # 实验名称 (Experiment name)
         #  n_parallel=args.num_workers_trpo, # Garage的并行采样器配置不同 (Garage's parallel sampler config is different)
         #  plot=True, # Garage的绘图通过其日志系统 (Garage's plotting via its logging system)
         #  snapshot_mode="all", # 由Snapshotter配置 (Configured by Snapshotter)
         #  use_cloudpickle=True # Garage默认使用cloudpickle (Garage uses cloudpickle by default)
         seed=args.seed # 传递种子给wrap_experiment (Pass seed to wrap_experiment)
         )
