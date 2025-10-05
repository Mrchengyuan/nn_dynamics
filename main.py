
# 导入 (imports)
import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import numpy.random as npr # 导入numpy的随机数生成模块 (Import numpy's random number generation module)
import torch # 导入torch库 (Import torch library)
import torch.nn as nn # 导入torch.nn模块，用于构建神经网络 (Import torch.nn module for building neural networks)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
import pickle # 导入pickle模块，用于序列化和反序列化Python对象 (Import pickle module for serializing and deserializing Python objects)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)
import os # 导入os模块，用于与操作系统交互 (Import os module for interacting with the operating system)
import sys # 导入sys模块，用于访问与Python解释器相关的变量和函数 (Import sys module for accessing Python interpreter related variables and functions)
from six.moves import cPickle # 从six.moves导入cPickle，用于Python 2/3兼容的pickle (Import cPickle from six.moves for Python 2/3 compatible pickle)
import yaml # 导入yaml模块，用于处理YAML文件 (Import yaml module for handling YAML files)
import argparse # 导入argparse模块，用于解析命令行参数 (Import argparse module for parsing command-line arguments)
import json # 导入json模块，用于处理JSON数据 (Import json module for handling JSON data)
import math # 导入math模块 (Import math module)

# 自定义模块导入 (my imports)
from policy_random import Policy_Random # 从policy_random模块导入Policy_Random类 (Import Policy_Random class from policy_random module)
from trajectories import make_trajectory # 从trajectories模块导入make_trajectory函数 (Import make_trajectory function from trajectories module)
from trajectories import get_trajfollow_params # 从trajectories模块导入get_trajfollow_params函数 (Import get_trajfollow_params function from trajectories module)
from data_manipulation import generate_training_data_inputs # 从data_manipulation模块导入generate_training_data_inputs函数 (Import generate_training_data_inputs function from data_manipulation module)
from data_manipulation import generate_training_data_outputs # 从data_manipulation模块导入generate_training_data_outputs函数 (Import generate_training_data_outputs function from data_manipulation module)
from data_manipulation import from_observation_to_usablestate # 从data_manipulation模块导入from_observation_to_usablestate函数 (Import from_observation_to_usablestate function from data_manipulation module)
from data_manipulation import get_indices # 从data_manipulation模块导入get_indices函数 (Import get_indices function from data_manipulation module)
from helper_funcs import perform_rollouts # 从helper_funcs模块导入perform_rollouts函数 (Import perform_rollouts function from helper_funcs module)
from helper_funcs import create_env # 从helper_funcs模块导入create_env函数 (Import create_env function from helper_funcs module)
from helper_funcs import visualize_rendering # 从helper_funcs模块导入visualize_rendering函数 (Import visualize_rendering function from helper_funcs module)
from helper_funcs import add_noise # 从helper_funcs模块导入add_noise函数 (Import add_noise function from helper_funcs module)
from dynamics_model import Dyn_Model # 从dynamics_model模块导入Dyn_Model类 (Import Dyn_Model class from dynamics_model module)
from mpc_controller import MPCController # 从mpc_controller模块导入MPCController类 (Import MPCController class from mpc_controller module)

def main(): # 定义主函数 (Define main function)

    #################################################
    ############ 命令行参数 (commandline arguments) ##############
    #################################################

    parser = argparse.ArgumentParser() # 创建一个ArgumentParser对象 (Create an ArgumentParser object)
    parser.add_argument('--yaml_file', type=str, default='ant_forward') # 添加yaml_file参数 (Add yaml_file argument)
    parser.add_argument('--seed', type=int, default=0) # 添加seed参数 (Add seed argument)
    parser.add_argument('--run_num', type=int, default=0) # 添加run_num参数 (Add run_num argument)
    parser.add_argument('--use_existing_training_data', action="store_true", dest='use_existing_training_data', default=False) # 添加use_existing_training_data参数 (Add use_existing_training_data argument)
    parser.add_argument('--use_existing_dynamics_model', action="store_true", dest='use_existing_dynamics_model', default=False) # 添加use_existing_dynamics_model参数 (Add use_existing_dynamics_model argument)

    parser.add_argument('--desired_traj_type', type=str, default='straight') # 添加desired_traj_type参数 (straight, left_turn, right_turn, u_turn, backward, forward_backward) (Add desired_traj_type argument)
    parser.add_argument('--num_rollouts_save_for_mf', type=int, default=60) # 添加num_rollouts_save_for_mf参数 (Add num_rollouts_save_for_mf argument)

    # 额外参数用于测试时覆盖yaml配置 (Extra parameters to override yaml config during tests)
    parser.add_argument('--num_aggregation_iters', type=int, default=None) # 聚合迭代次数 (Number of aggregation iterations)
    parser.add_argument('--num_rollouts_train', type=int, default=None) # 训练rollout数量 (Number of rollouts for training)
    parser.add_argument('--num_rollouts_val', type=int, default=None) # 验证rollout数量 (Number of rollouts for validation)
    parser.add_argument('--steps_per_rollout_train', type=int, default=None) # 训练每个rollout的步数 (Steps per rollout for training)
    parser.add_argument('--steps_per_rollout_val', type=int, default=None) # 验证每个rollout的步数 (Steps per rollout for validation)
    parser.add_argument('--nEpoch', type=int, default=None) # 训练周期数 (Number of epochs)
    parser.add_argument('--num_trajectories_for_aggregation', type=int, default=None) # 用于聚合的轨迹数量 (Number of trajectories for aggregation)
    parser.add_argument('--rollouts_forTraining', type=int, default=None) # 聚合训练用的rollout数量 (Number of rollouts used for training in aggregation)

    parser.add_argument('--might_render', action="store_true", dest='might_render', default=False) # 添加might_render参数 (Add might_render argument)
    parser.add_argument('--visualize_MPC_rollout', action="store_true", dest='visualize_MPC_rollout', default=False) # 添加visualize_MPC_rollout参数 (Add visualize_MPC_rollout argument)
    parser.add_argument('--perform_forwardsim_for_vis', action="store_true", dest='perform_forwardsim_for_vis', default=False) # 添加perform_forwardsim_for_vis参数 (Add perform_forwardsim_for_vis argument)
    parser.add_argument('--print_minimal', dest='print_minimal', nargs='?', const=True,
                        default=False, type=lambda x: str(x).lower() == 'true') # 添加print_minimal参数, 允许显式True/False (Add print_minimal argument, allow explicit True/False)
    args = parser.parse_args() # 解析命令行参数 (Parse command-line arguments)

    might_render = False # 初始化是否渲染标志 (Initialize flag whether to render)
    if args.visualize_MPC_rollout or args.might_render: # 如果需要可视化 (If visualization is requested)
        might_render = True # 设置标志为True (Set flag to True)


    ########################################
    ######### YAML文件中的参数 (params from yaml file) ########
    ########################################

    # 从指定文件加载参数 (load in parameters from specified file)
    yaml_path = os.path.abspath('yaml_files/'+args.yaml_file+'.yaml') # 获取YAML文件的绝对路径 (Get the absolute path of the YAML file)
    assert(os.path.exists(yaml_path)) # 确认YAML文件存在 (Assert that the YAML file exists)
    with open(yaml_path, 'r') as f: # 打开YAML文件 (Open the YAML file)
        params = yaml.safe_load(f) # 加载YAML文件内容 (Load the YAML file content using safe_load)

    # 保存从指定文件加载的参数 (save params from specified file)
    which_agent = params['which_agent'] # 智能体类型 (Agent type)
    follow_trajectories = params['follow_trajectories'] # 是否跟随轨迹 (Whether to follow trajectories)
    # 数据收集 (data collection)
    use_threading = params['data_collection']['use_threading'] # 是否使用多线程 (Whether to use multithreading)
    num_rollouts_train = params['data_collection']['num_rollouts_train'] # 训练用rollout数量 (Number of rollouts for training)
    num_rollouts_val = params['data_collection']['num_rollouts_val'] # 验证用rollout数量 (Number of rollouts for validation)
    # 动力学模型 (dynamics model)
    num_fc_layers = params['dyn_model']['num_fc_layers'] # 全连接层数量 (Number of fully connected layers)
    depth_fc_layers = params['dyn_model']['depth_fc_layers'] # 全连接层深度 (Depth of fully connected layers)
    batchsize = params['dyn_model']['batchsize'] # 批处理大小 (Batch size)
    lr = params['dyn_model']['lr'] # 学习率 (Learning rate)
    nEpoch = params['dyn_model']['nEpoch'] # 训练轮数 (Number of epochs)
    fraction_use_new = params['dyn_model']['fraction_use_new'] # 使用新数据的比例 (Fraction of new data to use)
    # 控制器 (controller)
    horizon = params['controller']['horizon'] # 控制时域 (Control horizon)
    num_control_samples = params['controller']['num_control_samples'] # 控制采样数量 (Number of control samples)
    if(which_agent==1): # 如果是蚂蚁智能体 (If it's the Ant agent)
        if(args.desired_traj_type=='straight'): # 如果期望轨迹类型是直线 (If the desired trajectory type is straight)
            num_control_samples=3000 # 设置控制采样数量为3000 (Set the number of control samples to 3000)
    # 聚合 (aggregation)
    num_aggregation_iters = params['aggregation']['num_aggregation_iters'] # 聚合迭代次数 (Number of aggregation iterations)
    num_trajectories_for_aggregation = params['aggregation']['num_trajectories_for_aggregation'] # 用于聚合的轨迹数量 (Number of trajectories for aggregation)
    rollouts_forTraining = params['aggregation']['rollouts_forTraining'] # 用于训练的rollout数量 (Number of rollouts for training)
    # 噪声 (noise)
    make_aggregated_dataset_noisy = params['noise']['make_aggregated_dataset_noisy'] # 是否对聚合数据集添加噪声 (Whether to add noise to the aggregated dataset)
    make_training_dataset_noisy = params['noise']['make_training_dataset_noisy'] # 是否对训练数据集添加噪声 (Whether to add noise to the training dataset)
    noise_actions_during_MPC_rollouts = params['noise']['noise_actions_during_MPC_rollouts'] # MPC rollout期间是否对动作添加噪声 (Whether to add noise to actions during MPC rollouts)
    # 步数 (steps)
    dt_steps = params['steps']['dt_steps'] # 时间步长 (Time step duration)
    steps_per_episode = params['steps']['steps_per_episode'] # 每回合步数 (Steps per episode)
    steps_per_rollout_train = params['steps']['steps_per_rollout_train'] # 训练用rollout的每回合步数 (Steps per rollout for training)
    steps_per_rollout_val = params['steps']['steps_per_rollout_val'] # 验证用rollout的每回合步数 (Steps per rollout for validation)
    # 保存 (saving)
    min_rew_for_saving = params['saving']['min_rew_for_saving'] # 保存的最小奖励阈值 (Minimum reward threshold for saving)
    # 通用 (generic)
    visualize_True = params['generic']['visualize_True'] # 可视化选项True (Visualize option True)
    visualize_False = params['generic']['visualize_False'] # 可视化选项False (Visualize option False)
    # 来自命令行参数 (from args)
    print_minimal = args.print_minimal # 是否最小化打印信息 (Whether to minimize print output)

    # 如提供命令行参数则覆盖YAML中的设置 (override YAML settings if provided via command line)
    if args.num_aggregation_iters is not None:
        num_aggregation_iters = args.num_aggregation_iters
    if args.num_rollouts_train is not None:
        num_rollouts_train = args.num_rollouts_train
    if args.num_rollouts_val is not None:
        num_rollouts_val = args.num_rollouts_val
    if args.steps_per_rollout_train is not None:
        steps_per_rollout_train = args.steps_per_rollout_train
    if args.steps_per_rollout_val is not None:
        steps_per_rollout_val = args.steps_per_rollout_val
    if args.nEpoch is not None:
        nEpoch = args.nEpoch
    if args.num_trajectories_for_aggregation is not None:
        num_trajectories_for_aggregation = args.num_trajectories_for_aggregation
    if args.rollouts_forTraining is not None:
        rollouts_forTraining = args.rollouts_forTraining


    ########################################
    ### 创建用于保存数据的目录 (make directories for saving data) ###
    ########################################

    save_dir = 'run_'+ str(args.run_num) # 定义保存目录名称 (Define save directory name)
    if not os.path.exists(save_dir): # 如果目录不存在 (If directory does not exist)
        os.makedirs(save_dir) # 创建主保存目录 (Create main save directory)
        os.makedirs(save_dir+'/losses') # 创建损失保存目录 (Create losses save directory)
        os.makedirs(save_dir+'/models') # 创建模型保存目录 (Create models save directory)
        os.makedirs(save_dir+'/saved_forwardsim') # 创建前向模拟保存目录 (Create forward simulation save directory)
        os.makedirs(save_dir+'/saved_trajfollow') # 创建轨迹跟随保存目录 (Create trajectory following save directory)
        os.makedirs(save_dir+'/training_data') # 创建训练数据保存目录 (Create training data save directory)

    ########################################
    ############## 设置变量 (set vars) ################
    ########################################

    # 设置随机种子 (set seeds)
    npr.seed(args.seed) # 设置numpy的随机种子 (Set numpy's random seed)
    torch.manual_seed(args.seed) # 设置torch的CPU随机种子 (Set torch's CPU random seed)
    if torch.cuda.is_available(): # 如果CUDA可用 (If CUDA is available)
        torch.cuda.manual_seed_all(args.seed) # 设置torch所有GPU的随机种子 (Set torch's random seed for all GPUs)

    # 确定设备 (Determine device)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): # 如果MPS可用且已构建 (If MPS is available and built)
        device = torch.device("mps") # 使用MPS设备 (Use MPS device)
    elif torch.cuda.is_available(): # 否则如果CUDA可用 (Else if CUDA is available)
        device = torch.device("cuda") # 使用CUDA设备 (Use CUDA device)
    else: # 否则 (Otherwise)
        device = torch.device("cpu") # 使用CPU设备 (Use CPU device)
    if not print_minimal: # 如果不是最小化打印 (If not minimal print)
        print(f"Using device: {device}") # 打印正在使用的设备 (Print the device being used)
    
    # 数据收集，带或不带多线程 (data collection, either with or without multi-threading)
    if(use_threading): # 如果使用多线程 (If using multi-threading)
        from collect_samples_threaded import CollectSamples # 从collect_samples_threaded导入CollectSamples (Import CollectSamples from collect_samples_threaded)
    else: # 否则 (Otherwise)
        from collect_samples import CollectSamples # 从collect_samples导入CollectSamples (Import CollectSamples from collect_samples)

    # 更多变量 (more vars)
    x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(which_agent) # 获取智能体的索引 (Get indices for the agent)
    # tf_datatype = tf.float64 # TensorFlow数据类型 (TensorFlow data type) -> PyTorch使用torch.double (PyTorch uses torch.double)
    torch_datatype = torch.float32 # PyTorch数据类型 (PyTorch data type)
    noiseToSignal = 0.01 # 噪声信号比 (Noise to signal ratio)

    # n代表噪声，c代表干净... 第一个字母表示执行的动作，第二个字母表示聚合的动作 (n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated)
    actions_ag='nc' # 聚合动作的类型 (Type of aggregated actions)
    
    #################################################
    ######## 将参数值保存到文件 (save param values to a file) ############
    #################################################

    param_dict={} # 初始化参数字典 (Initialize parameter dictionary)
    param_dict['which_agent']= which_agent # 智能体类型 (Agent type)
    # ... (其余参数的保存逻辑与原代码类似) ... (The saving logic for the rest of the parameters is similar to the original code)
    param_dict['use_existing_training_data']= str(args.use_existing_training_data)
    param_dict['desired_traj_type']= args.desired_traj_type
    param_dict['visualize_MPC_rollout']= str(args.visualize_MPC_rollout)
    param_dict['num_rollouts_save_for_mf']= args.num_rollouts_save_for_mf
    param_dict['seed']= args.seed
    param_dict['follow_trajectories']= str(follow_trajectories)
    param_dict['use_threading']= str(use_threading)
    param_dict['num_rollouts_train']= num_rollouts_train
    param_dict['num_fc_layers']= num_fc_layers
    param_dict['depth_fc_layers']= depth_fc_layers
    param_dict['batchsize']= batchsize
    param_dict['lr']= lr
    param_dict['nEpoch']= nEpoch
    param_dict['fraction_use_new']= fraction_use_new
    param_dict['horizon']= horizon
    param_dict['num_control_samples']= num_control_samples
    param_dict['num_aggregation_iters']= num_aggregation_iters
    param_dict['num_trajectories_for_aggregation']= num_trajectories_for_aggregation
    param_dict['rollouts_forTraining']= rollouts_forTraining
    param_dict['make_aggregated_dataset_noisy']= str(make_aggregated_dataset_noisy)
    param_dict['make_training_dataset_noisy']= str(make_training_dataset_noisy)
    param_dict['noise_actions_during_MPC_rollouts']= str(noise_actions_during_MPC_rollouts)
    param_dict['dt_steps']= dt_steps
    param_dict['steps_per_episode']= steps_per_episode
    param_dict['steps_per_rollout_train']= steps_per_rollout_train
    param_dict['steps_per_rollout_val']= steps_per_rollout_val
    param_dict['min_rew_for_saving']= min_rew_for_saving
    param_dict['x_index']= x_index
    param_dict['y_index']= y_index
    param_dict['torch_datatype']= str(torch_datatype) # 保存PyTorch数据类型 (Save PyTorch data type)
    param_dict['noiseToSignal']= noiseToSignal

    with open(save_dir+'/params.pkl', 'wb') as f: # 打开参数的pickle文件 (Open pickle file for parameters)
        pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL) # 保存参数字典 (Save parameter dictionary)
    with open(save_dir+'/params.txt', 'w') as f: # 打开参数的文本文件 (Open text file for parameters)
        f.write(json.dumps(param_dict)) # 以JSON格式写入参数 (Write parameters in JSON format)

    #################################################
    ### 初始化实验 (initialize the experiment)
    #################################################

    if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
        print("\n#####################################") # 打印分隔符 (Print separator)
        print("Initializing environment") # 打印初始化环境信息 (Print initializing environment message)
        print("#####################################\n") # 打印分隔符 (Print separator)

    # 创建环境 (create env)
    env, dt_from_xml = create_env(which_agent, render_mode='human' if might_render else None) # 创建环境并根据需要设置渲染模式 (Create environment and set render mode if needed)
    # 注意: create_env可能包含TensorFlow依赖，需要检查 (Note: create_env might contain TensorFlow dependencies, needs checking)

    # 为数据收集创建随机策略 (create random policy for data collection)
    random_policy = Policy_Random(env) # 创建Policy_Random实例 (Create Policy_Random instance)

    # TensorFlow的GPU选项和会话在此处移除 (TensorFlow GPU options and session are removed here)
    # PyTorch的设备管理已在前面完成 (PyTorch device management was done earlier)

    #################################################
    ### 处理数据 (deal with data)
    #################################################

    if(args.use_existing_training_data): # 如果使用现有的训练数据 (If using existing training data)
        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Retrieving training data & policy from saved files") # 打印从保存文件中检索训练数据和策略的信息 (Print message about retrieving training data and policy from saved files)
            print("#####################################\n") # 打印分隔符 (Print separator)

        dataX = np.load(save_dir + '/training_data/dataX.npy') # 加载dataX (Load dataX) (input1: state)
        dataY = np.load(save_dir + '/training_data/dataY.npy') # 加载dataY (Load dataY) (input2: control)
        dataZ = np.load(save_dir + '/training_data/dataZ.npy') # 加载dataZ (Load dataZ) (output: nextstate-state)
        states_val = np.load(save_dir + '/training_data/states_val.npy') # 加载验证状态 (Load validation states)
        controls_val = np.load(save_dir + '/training_data/controls_val.npy') # 加载验证控制 (Load validation controls)
        forwardsim_x_true = np.load(save_dir + '/training_data/forwardsim_x_true.npy') # 加载前向模拟的真实x (Load true x for forward simulation)
        forwardsim_y = np.load(save_dir + '/training_data/forwardsim_y.npy') # 加载前向模拟的y (Load y for forward simulation)

    else: # 否则 (Otherwise)
        # ... (数据收集逻辑与原代码类似，但确保与PyTorch兼容) ... (Data collection logic similar to original, but ensure PyTorch compatibility)
        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Performing rollouts to collect training data") # 打印执行rollout收集训练数据的信息 (Print message about performing rollouts to collect training data)
            print("#####################################\n") # 打印分隔符 (Print separator)

        # 执行rollouts (perform rollouts)
        states, controls, _, _ = perform_rollouts(random_policy, num_rollouts_train, steps_per_rollout_train, visualize_False,
                                                CollectSamples, env, which_agent, dt_steps, dt_from_xml, follow_trajectories) # 执行训练数据的rollouts (Perform rollouts for training data)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Performing rollouts to collect validation data") # 打印执行rollout收集验证数据的信息 (Print message about performing rollouts to collect validation data)
            print("#####################################\n") # 打印分隔符 (Print separator)

        start_validation_rollouts = time.time() # 记录验证rollout开始时间 (Record validation rollout start time)
        states_val, controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val, steps_per_rollout_val, visualize_False,
                                                        CollectSamples, env, which_agent, dt_steps, dt_from_xml, follow_trajectories) # 执行验证数据的rollouts (Perform rollouts for validation data)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Convert from env observations to NN 'states' ") # 打印将环境观测转换为神经网络状态的信息 (Print message about converting environment observations to neural network states)
            print("#####################################\n") # 打印分隔符 (Print separator)

        # 训练 (training)
        states = from_observation_to_usablestate(states, which_agent, False) # 转换训练状态 (Convert training states)
        # 验证 (validation)
        states_val = from_observation_to_usablestate(states_val, which_agent, False) # 转换验证状态 (Convert validation states)
        states_val = np.array(states_val) # 将验证状态转换为numpy数组 (Convert validation states to numpy array)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Data formatting: create inputs and labels for NN ") # 打印数据格式化信息 (Print message about data formatting)
            print("#####################################\n") # 打印分隔符 (Print separator)

        dataX , dataY = generate_training_data_inputs(states, controls) # 生成训练输入数据 (Generate training input data)
        dataZ = generate_training_data_outputs(states, which_agent) # 生成训练输出数据 (Generate training output data)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Add noise") # 打印添加噪声信息 (Print message about adding noise)
            print("#####################################\n") # 打印分隔符 (Print separator)

        # 添加少量动力学噪声 (add a little dynamics noise) (next state is not perfectly accurate, given correct state and action)
        if(make_training_dataset_noisy): # 如果对训练数据集添加噪声 (If adding noise to training dataset)
            dataX = add_noise(dataX, noiseToSignal) # 对dataX添加噪声 (Add noise to dataX)
            dataZ = add_noise(dataZ, noiseToSignal) # 对dataZ添加噪声 (Add noise to dataZ)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Perform rollout & save for forward sim") # 打印执行rollout并保存用于前向模拟的信息 (Print message about performing rollout and saving for forward simulation)
            print("#####################################\n") # 打印分隔符 (Print separator)

        states_forwardsim_orig, controls_forwardsim, _,_ = perform_rollouts(random_policy, 1, 100,
                                                                        visualize_False, CollectSamples,
                                                                        env, which_agent, dt_steps,
                                                                        dt_from_xml, follow_trajectories) # 执行用于前向模拟的rollout (Perform rollout for forward simulation)
        states_forwardsim = np.copy(from_observation_to_usablestate(states_forwardsim_orig, which_agent, False)) # 转换前向模拟的状态 (Convert states for forward simulation)
        forwardsim_x_true, forwardsim_y = generate_training_data_inputs(states_forwardsim, controls_forwardsim) # 生成前向模拟的输入 (Generate inputs for forward simulation)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Saving data") # 打印保存数据信息 (Print message about saving data)
            print("#####################################\n") # 打印分隔符 (Print separator)

        np.save(save_dir + '/training_data/dataX.npy', dataX) # 保存dataX (Save dataX)
        np.save(save_dir + '/training_data/dataY.npy', dataY) # 保存dataY (Save dataY)
        np.save(save_dir + '/training_data/dataZ.npy', dataZ) # 保存dataZ (Save dataZ)
        np.save(save_dir + '/training_data/states_val.npy', states_val) # 保存验证状态 (Save validation states)
        np.save(save_dir + '/training_data/controls_val.npy', controls_val) # 保存验证控制 (Save validation controls)
        np.save(save_dir + '/training_data/forwardsim_x_true.npy', forwardsim_x_true) # 保存前向模拟的真实x (Save true x for forward simulation)
        np.save(save_dir + '/training_data/forwardsim_y.npy', forwardsim_y) # 保存前向模拟的y (Save y for forward simulation)

    if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
        print("Done getting data.") # 打印数据获取完成信息 (Print message that data collection is done)
        print("dataX dim: ", dataX.shape) # 打印dataX维度 (Print dimension of dataX)

    #################################################
    ### 初始化变量 (init vars)
    #################################################

    counter_agg_iters=0 # 聚合迭代计数器 (Aggregation iteration counter)
    training_loss_list=[] # 训练损失列表 (Training loss list)
    # ... (其余列表初始化与原代码类似) ... (Rest of the list initializations are similar to original code)
    old_loss_list=[]
    new_loss_list=[]
    errors_1_per_agg=[]
    errors_5_per_agg=[]
    errors_10_per_agg=[]
    errors_50_per_agg=[]
    errors_100_per_agg=[]
    list_avg_rew=[]
    list_num_datapoints=[]
    dataX_new = np.zeros((0,dataX.shape[1])) # 初始化新的dataX (Initialize new dataX)
    dataY_new = np.zeros((0,dataY.shape[1])) # 初始化新的dataY (Initialize new dataY)
    dataZ_new = np.zeros((0,dataZ.shape[1])) # 初始化新的dataZ (Initialize new dataZ)

    #################################################
    ### 预处理旧的训练数据集 (preprocess the old training dataset)
    #################################################

    if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
        print("\n#####################################") # 打印分隔符 (Print separator)
        print("Preprocessing 'old' training data") # 打印预处理旧训练数据信息 (Print message about preprocessing old training data)
        print("#####################################\n") # 打印分隔符 (Print separator)

    # 每个分量（例如x位置）都应变为均值0，标准差1 (every component (i.e. x position) should become mean 0, std 1)
    mean_x = np.mean(dataX, axis = 0) # 计算dataX的均值 (Calculate mean of dataX)
    dataX_normalized = dataX - mean_x # dataX减去均值 (Subtract mean from dataX)
    std_x = np.std(dataX_normalized, axis = 0) # 计算标准化后dataX的标准差 (Calculate std of normalized dataX)
    dataX_normalized = np.nan_to_num(dataX_normalized/std_x) # dataX除以标准差，并将NaN替换为0 (Divide dataX by std and replace NaN with 0)

    mean_y = np.mean(dataY, axis = 0)  # 计算dataY的均值 (Calculate mean of dataY)
    dataY_normalized = dataY - mean_y # dataY减去均值 (Subtract mean from dataY)
    std_y = np.std(dataY_normalized, axis = 0) # 计算标准化后dataY的标准差 (Calculate std of normalized dataY)
    dataY_normalized = np.nan_to_num(dataY_normalized/std_y) # dataY除以标准差 (Divide dataY by std)

    mean_z = np.mean(dataZ, axis = 0) # 计算dataZ的均值 (Calculate mean of dataZ)
    dataZ_normalized = dataZ - mean_z # dataZ减去均值 (Subtract mean from dataZ)
    std_z = np.std(dataZ_normalized, axis = 0) # 计算标准化后dataZ的标准差 (Calculate std of normalized dataZ)
    dataZ_normalized = np.nan_to_num(dataZ_normalized/std_z) # dataZ除以标准差 (Divide dataZ by std)

    ## 连接状态和动作，用于训练动力学模型 (concatenate state and action, to be used for training dynamics)
    inputs_normalized = np.concatenate((dataX_normalized, dataY_normalized), axis=1) # 连接标准化的dataX和dataY (Concatenate normalized dataX and dataY)
    outputs_normalized = np.copy(dataZ_normalized) # 复制标准化的dataZ (Copy normalized dataZ)

    # 此处的渲染调用是为了避免稍后出现错误 (doing a render here somehow allows it to not produce an error later)
    if(might_render): # 如果需要渲染 (If rendering is desired)
        new_env, _ = create_env(which_agent, render_mode='human') # 创建带渲染的环境 (Create environment with rendering)
        new_env.render() # 渲染环境 (Render environment)
        new_env.close() # 关闭环境 (Close environment)

    ##############################################
    ########## 聚合循环 (THE AGGREGATION LOOP) ##############
    ##############################################

    # 维度 (dimensions)
    assert inputs_normalized.shape[0] == outputs_normalized.shape[0] # 确认输入和输出的样本数相同 (Assert that input and output have the same number of samples)
    inputSize = inputs_normalized.shape[1] # 输入大小 (Input size)
    outputSize = outputs_normalized.shape[1] # 输出大小 (Output size)

    # 初始化动力学模型 (initialize dynamics model)
    # 注意：TensorFlow会话参数已移除 (Note: TensorFlow session parameter is removed)
    dyn_model = Dyn_Model(inputSize, outputSize, lr, batchsize, which_agent, x_index, y_index, num_fc_layers,
                        depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, print_minimal, device=device) # 创建Dyn_Model实例 (Create Dyn_Model instance)
    dyn_model = dyn_model.to(dtype=torch_datatype, device=device) # 设置模型的数据类型和设备 (Set model dtype and device)


    # 创建MPC控制器 (create mpc controller)
    # 注意: MPCController可能需要更新以完全兼容PyTorch Dyn_Model (Note: MPCController might need updates for full PyTorch Dyn_Model compatibility)
    mpc_controller = MPCController(env, dyn_model, horizon, which_agent, steps_per_episode, dt_steps, num_control_samples,
                                    mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index,
                                    z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index,
                                    frontfoot_index, xvel_index, orientation_index, torch_device=device, torch_dtype=torch_datatype) # 创建MPCController实例 (Create MPCController instance)

    # TensorFlow的变量初始化已移除 (TensorFlow variable initialization is removed)

    while(counter_agg_iters < num_aggregation_iters): # 当聚合迭代次数小于总迭代次数时 (While aggregation iteration count is less than total iterations)

        # PyTorch中模型保存方式不同，tf.train.Saver移除 (Model saving is different in PyTorch, tf.train.Saver removed)
        # 模型保存将在训练后使用dyn_model.save_weights() (Model saving will use dyn_model.save_weights() after training)

        print("\n#####################################") # 打印分隔符 (Print separator)
        print("AGGREGATION ITERATION ", counter_agg_iters) # 打印当前聚合迭代次数 (Print current aggregation iteration)
        print("#####################################\n") # 打印分隔符 (Print separator)

        # 保存此聚合迭代中用于训练的聚合数据集 (save the aggregated dataset used to train during this agg iteration)
        np.save(save_dir + '/training_data/dataX_new_iter'+ str(counter_agg_iters) + '.npy', dataX_new) # 保存新的dataX (Save new dataX)
        np.save(save_dir + '/training_data/dataY_new_iter'+ str(counter_agg_iters) + '.npy', dataY_new) # 保存新的dataY (Save new dataY)
        np.save(save_dir + '/training_data/dataZ_new_iter'+ str(counter_agg_iters) + '.npy', dataZ_new) # 保存新的dataZ (Save new dataZ)

        starting_big_loop = time.time() # 记录大循环开始时间 (Record start time of the big loop)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Preprocessing 'new' training data") # 打印预处理新训练数据信息 (Print message about preprocessing new training data)
            print("#####################################\n") # 打印分隔符 (Print separator)

        # 预处理新的训练数据 (Preprocessing 'new' training data)
        dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x) / (std_x + 1e-8)) # 预处理新的dataX (Preprocess new dataX, add epsilon to std)
        dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y) / (std_y + 1e-8)) # 预处理新的dataY (Preprocess new dataY, add epsilon to std)
        dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z) / (std_z + 1e-8)) # 预处理新的dataZ (Preprocess new dataZ, add epsilon to std)

        ## 连接状态和动作，用于训练动力学 (concatenate state and action, to be used for training dynamics)
        inputs_new_normalized = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1) # 连接标准化的新输入 (Concatenate normalized new inputs)
        outputs_new_normalized = np.copy(dataZ_new_preprocessed) # 复制标准化的新输出 (Copy normalized new outputs)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Training the dynamics model") # 打印训练动力学模型信息 (Print message about training the dynamics model)
            print("#####################################\n") # 打印分隔符 (Print separator)

        # 训练模型或恢复模型 (train model or restore model)
        if(args.use_existing_dynamics_model and counter_agg_iters == 0): # 如果使用现有动力学模型且是第一次聚合迭代 (If using existing dynamics model and it's the first aggregation iteration)
            restore_path = save_dir+ '/models/finalModel.pth' # 定义恢复路径 (Define restore path with .pth extension)
            if os.path.exists(restore_path): # 如果路径存在 (If path exists)
                dyn_model.load_weights(restore_path) # 加载模型权重 (Load model weights)
                # training_loss, old_loss, new_loss 将不会被计算，因为模型已加载 (training_loss, old_loss, new_loss will not be calculated as model is loaded)
                current_training_loss_epoch_avg = 0 # 当前训练损失周期平均值 (Current training loss epoch average)
                current_old_loss_epoch_avg = 0 # 当前旧损失周期平均值 (Current old loss epoch average)
                current_new_loss_epoch_avg = 0 # 当前新损失周期平均值 (Current new loss epoch average)
                print(f"Model restored from {restore_path}") # 打印模型已从路径恢复的信息 (Print message that model was restored from path)
            else: # 否则 (Otherwise)
                print(f"Warning: Tried to load existing model from {restore_path}, but file not found. Training from scratch.") # 打印警告信息 (Print warning message)
                args.use_existing_dynamics_model = False # 设置use_existing_dynamics_model为False (Set use_existing_dynamics_model to False)

        if not args.use_existing_dynamics_model or counter_agg_iters > 0 or not os.path.exists(save_dir+ '/models/finalModel.pth'): # 如果不使用现有模型，或不是第一次迭代，或模型文件不存在 (If not using existing model, or not first iteration, or model file doesn't exist)
            # 准备合并的数据集 (Prepare combined dataset for training)
            combined_inputs = np.concatenate((inputs_normalized, inputs_new_normalized), axis=0) if dataX_new.shape[0] > 0 else inputs_normalized # 合并的输入 (Combined inputs)
            combined_outputs = np.concatenate((outputs_normalized, outputs_new_normalized), axis=0) if dataX_new.shape[0] > 0 else outputs_normalized # 合并的输出 (Combined outputs)
            
            epoch_training_losses = [] # 初始化周期训练损失列表 (Initialize epoch training losses list)
            for epoch_num in range(nEpoch): # 对于每个训练轮数 (For each epoch)
                epoch_loss = 0 # 初始化周期损失 (Initialize epoch loss)
                num_batches = 0 # 初始化批次数 (Initialize number of batches)

                # 创建索引并打乱 (Create indices and shuffle)
                num_total_samples = combined_inputs.shape[0] # 总样本数 (Total number of samples)
                shuffled_indices = npr.permutation(num_total_samples) # 打乱的索引 (Shuffled indices)

                for batch_idx in range(0, num_total_samples, batchsize): # 对于每个批次索引 (For each batch index)
                    batch_indices = shuffled_indices[batch_idx : batch_idx + batchsize] # 获取批次索引 (Get batch indices)
                    batch_X = combined_inputs[batch_indices] # 获取批次输入X (Get batch input X)
                    batch_Y = combined_outputs[batch_indices] # 获取批次输入Y (Get batch input Y)

                    loss_val = dyn_model.train_step(batch_X, batch_Y) # 执行单个训练步骤 (Perform a single training step)
                    epoch_loss += loss_val # 累加损失值 (Accumulate loss value)
                    num_batches += 1 # 批次数加1 (Increment batch count)

                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0 # 计算平均周期损失 (Calculate average epoch loss)
                epoch_training_losses.append(avg_epoch_loss) # 添加平均周期损失到列表 (Add average epoch loss to list)
                if not print_minimal and (epoch_num % 10) == 0: # 如果不是最小化打印且周期数是10的倍数 (If not minimal print and epoch number is a multiple of 10)
                    print(f"Aggregation iter {counter_agg_iters}, Epoch {epoch_num}, Avg Loss: {avg_epoch_loss:.6f}") # 打印聚合迭代次数、周期数和平均损失 (Print aggregation iteration, epoch number, and average loss)
            
            current_training_loss_epoch_avg = np.mean(epoch_training_losses) if epoch_training_losses else 0 # 计算当前训练损失周期平均值 (Calculate current training loss epoch average)
            np.save(save_dir + f'/losses/training_losses_agg_iter_{counter_agg_iters}.npy', np.array(epoch_training_losses)) # 保存周期训练损失 (Save epoch training losses)

            # 在旧数据集上评估损失 (Evaluate loss on old dataset)
            current_old_loss_epoch_avg = dyn_model.run_validation(inputs_normalized, outputs_normalized) if inputs_normalized.shape[0] > 0 else 0 # 计算旧数据集上的损失 (Calculate loss on old dataset)
            # 在新数据集上评估损失 (Evaluate loss on new dataset)
            current_new_loss_epoch_avg = dyn_model.run_validation(inputs_new_normalized, outputs_new_normalized) if inputs_new_normalized.shape[0] > 0 else 0 # 计算新数据集上的损失 (Calculate loss on new dataset)

        # 记录模型的训练情况 (Log model training performance)
        training_loss_list.append(current_training_loss_epoch_avg) # 添加当前训练损失周期平均值 (Add current training loss epoch average)
        old_loss_list.append(current_old_loss_epoch_avg) # 添加当前旧损失周期平均值 (Add current old loss epoch average)
        new_loss_list.append(current_new_loss_epoch_avg) # 添加当前新损失周期平均值 (Add current new loss epoch average)

        print("\nAgg Iteration Training loss (avg over epochs): ", current_training_loss_epoch_avg) # 打印聚合迭代训练损失 (Print aggregation iteration training loss)

        #####################################
        ## 保存模型 (Saving model)
        #####################################
        dyn_model.save_weights(save_dir+ '/models/model_aggIter' +str(counter_agg_iters)+ '.pth') # 保存当前聚合迭代的模型权重 (Save model weights for current aggregation iteration)
        dyn_model.save_weights(save_dir+ '/models/finalModel.pth') # 保存最终模型权重 (Save final model weights)
        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print(f"Model saved at {save_dir}/models/model_aggIter{str(counter_agg_iters)}.pth") # 打印模型保存路径 (Print model save path)


        # ... (验证指标计算与原代码类似，但使用PyTorch张量和模型) ... (Validation metrics calculation similar to original, but using PyTorch tensors and model)
        if not print_minimal: # 如果不是最小化打印 (If not minimal print)
            print("\n#####################################") # 打印分隔符 (Print separator)
            print("Calculating Validation Metrics") # 打印计算验证指标信息 (Print message about calculating validation metrics)
            print("#####################################\n") # 打印分隔符 (Print separator)

        validation_inputs_states_np = [] # 初始化验证输入状态的numpy列表 (Initialize numpy list for validation input states)
        labels_1step_np = [] # 初始化1步标签的numpy列表 (Initialize numpy list for 1-step labels)
        labels_5step_np = [] # 初始化5步标签的numpy列表 (Initialize numpy list for 5-step labels)
        labels_10step_np = [] # 初始化10步标签的numpy列表 (Initialize numpy list for 10-step labels)
        labels_50step_np = [] # 初始化50步标签的numpy列表 (Initialize numpy list for 50-step labels)
        labels_100step_np = [] # 初始化100步标签的numpy列表 (Initialize numpy list for 100-step labels)
        controls_100step_np = [] # 初始化100步控制的numpy列表 (Initialize numpy list for 100-step controls)

        for i in range(len(states_val)): # 对于每个验证rollout (For each validation rollout)
            length_curr_rollout = states_val[i].shape[0] # 当前rollout的长度 (Length of current rollout)
            if length_curr_rollout > 100: # 如果长度大于100 (If length is greater than 100)
                validation_inputs_states_np.append(states_val[i][0:length_curr_rollout-100]) # 添加状态输入 (Add state inputs)
                list_100 = [] # 初始化100步控制列表 (Initialize list for 100-step controls)
                for j in range(100): # 对于100步中的每一步 (For each step in 100 steps)
                    list_100.append(controls_val[i][0+j:length_curr_rollout-100+j]) # 添加控制 (Add controls)
                list_100 = np.array(list_100) # 转换为numpy数组 (Convert to numpy array)
                list_100 = np.swapaxes(list_100,0,1) # 交换轴 (Swap axes)
                controls_100step_np.append(list_100) # 添加到100步控制列表 (Add to 100-step controls list)
                labels_1step_np.append(states_val[i][0+1:length_curr_rollout-100+1]) # 添加1步标签 (Add 1-step labels)
                labels_5step_np.append(states_val[i][0+5:length_curr_rollout-100+5]) # 添加5步标签 (Add 5-step labels)
                labels_10step_np.append(states_val[i][0+10:length_curr_rollout-100+10]) # 添加10步标签 (Add 10-step labels)
                labels_50step_np.append(states_val[i][0+50:length_curr_rollout-100+50]) # 添加50步标签 (Add 50-step labels)
                labels_100step_np.append(states_val[i][0+100:length_curr_rollout-100+100]) # 添加100步标签 (Add 100-step labels)

        if len(validation_inputs_states_np) == 0:
            if not print_minimal:  # 如果未收集到足够长的rollout，则打印提示并跳过 (Print notice if rollouts are too short)
                print("Skipping validation metrics due to short rollouts")
            return

        validation_inputs_states_np = np.concatenate(validation_inputs_states_np) # 连接验证输入状态 (Concatenate validation input states)
        controls_100step_np = np.concatenate(controls_100step_np) # 连接100步控制 (Concatenate 100-step controls)
        labels_1step_np = np.concatenate(labels_1step_np) # 连接1步标签 (Concatenate 1-step labels)
        labels_5step_np = np.concatenate(labels_5step_np) # 连接5步标签 (Concatenate 5-step labels)
        labels_10step_np = np.concatenate(labels_10step_np) # 连接10步标签 (Concatenate 10-step labels)
        labels_50step_np = np.concatenate(labels_50step_np) # 连接50步标签 (Concatenate 50-step labels)
        labels_100step_np = np.concatenate(labels_100step_np) # 连接100步标签 (Concatenate 100-step labels)
        
        # 转换为PyTorch张量以进行前向模拟 (Convert to PyTorch tensors for forward simulation)
        validation_inputs_states_torch = torch.from_numpy(validation_inputs_states_np).to(dtype=torch_datatype, device=device) # 验证输入状态的torch张量 (Torch tensor for validation input states)
        controls_100step_torch = torch.from_numpy(controls_100step_np).to(dtype=torch_datatype, device=device) # 100步控制的torch张量 (Torch tensor for 100-step controls)

        many_in_parallel = True # 并行处理标志 (Flag for parallel processing)
        # 注意：dyn_model.do_forward_sim现在接收PyTorch张量 (Note: dyn_model.do_forward_sim now expects PyTorch tensors)
        predicted_100step_list = dyn_model.do_forward_sim(validation_inputs_states_torch, controls_100step_torch,
                                                    many_in_parallel) # 执行前向模拟 (Perform forward simulation)
        predicted_100step = np.array([p_item for p_item in predicted_100step_list]) # 将预测结果转换为numpy数组 (Convert predictions to numpy array)


        # 转换为torch张量以计算误差 (Convert to torch tensors for error calculation)
        mean_x_torch = torch.from_numpy(mean_x).to(dtype=torch_datatype, device=device) # x均值的torch张量 (Torch tensor for mean_x)
        std_x_torch = torch.from_numpy(std_x).to(dtype=torch_datatype, device=device) # x标准差的torch张量 (Torch tensor for std_x)

        # 确保预测和标签具有相同的设备和数据类型 (Ensure predictions and labels have same device and dtype)
        predicted_100step_torch = torch.from_numpy(predicted_100step).to(dtype=torch_datatype, device=device) # 预测的100步torch张量 (Torch tensor for predicted 100 steps)
        labels_1step_torch = torch.from_numpy(labels_1step_np).to(dtype=torch_datatype, device=device) # 1步标签的torch张量 (Torch tensor for 1-step labels)
        labels_5step_torch = torch.from_numpy(labels_5step_np).to(dtype=torch_datatype, device=device) # 5步标签的torch张量 (Torch tensor for 5-step labels)
        labels_10step_torch = torch.from_numpy(labels_10step_np).to(dtype=torch_datatype, device=device) # 10步标签的torch张量 (Torch tensor for 10-step labels)
        labels_50step_torch = torch.from_numpy(labels_50step_np).to(dtype=torch_datatype, device=device) # 50步标签的torch张量 (Torch tensor for 50-step labels)
        labels_100step_torch = torch.from_numpy(labels_100step_np).to(dtype=torch_datatype, device=device) # 100步标签的torch张量 (Torch tensor for 100-step labels)

        array_meanx_torch = mean_x_torch.unsqueeze(0).repeat(labels_1step_torch.shape[0], 1) # x均值数组的torch张量 (Torch tensor for mean_x array)
        array_stdx_torch = std_x_torch.unsqueeze(0).repeat(labels_1step_torch.shape[0], 1) # x标准差数组的torch张量 (Torch tensor for std_x array)

        # 计算误差 (Calculate errors)
        error_1step = torch.mean(torch.square(torch.nan_to_num((predicted_100step_torch[1] - array_meanx_torch) / (array_stdx_torch + 1e-8))
                                - torch.nan_to_num((labels_1step_torch - array_meanx_torch) / (array_stdx_torch + 1e-8)))).item() # 1步误差 (1-step error)
        error_5step = torch.mean(torch.square(torch.nan_to_num((predicted_100step_torch[5] - array_meanx_torch) / (array_stdx_torch + 1e-8))
                                - torch.nan_to_num((labels_5step_torch - array_meanx_torch) / (array_stdx_torch + 1e-8)))).item() # 5步误差 (5-step error)
        error_10step = torch.mean(torch.square(torch.nan_to_num((predicted_100step_torch[10] - array_meanx_torch) / (array_stdx_torch + 1e-8))
                                 - torch.nan_to_num((labels_10step_torch - array_meanx_torch) / (array_stdx_torch + 1e-8)))).item() # 10步误差 (10-step error)
        error_50step = torch.mean(torch.square(torch.nan_to_num((predicted_100step_torch[50] - array_meanx_torch) / (array_stdx_torch + 1e-8))
                                 - torch.nan_to_num((labels_50step_torch - array_meanx_torch) / (array_stdx_torch + 1e-8)))).item() # 50步误差 (50-step error)
        error_100step = torch.mean(torch.square(torch.nan_to_num((predicted_100step_torch[100] - array_meanx_torch) / (array_stdx_torch + 1e-8))
                                  - torch.nan_to_num((labels_100step_torch - array_meanx_torch) / (array_stdx_torch + 1e-8)))).item() # 100步误差 (100-step error)
        print("Multistep error values: ", error_1step, error_5step, error_10step, error_50step, error_100step,"\n") # 打印多步误差值 (Print multistep error values)

        errors_1_per_agg.append(error_1step) # 添加1步误差 (Add 1-step error)
        errors_5_per_agg.append(error_5step) # 添加5步误差 (Add 5-step error)
        errors_10_per_agg.append(error_10step) # 添加10步误差 (Add 10-step error)
        errors_50_per_agg.append(error_50step) # 添加50步误差 (Add 50-step error)
        errors_100_per_agg.append(error_100step) # 添加100步误差 (Add 100-step error)

        if(args.perform_forwardsim_for_vis): # 如果执行用于可视化的前向模拟 (If performing forward simulation for visualization)
            if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
                print("\n#####################################") # 打印分隔符 (Print separator)
                print("Performing a forward sim of the learned model. using pre-saved dataset. just for visualization") # 打印执行前向模拟信息 (Print message about performing forward simulation)
                print("#####################################\n") # 打印分隔符 (Print separator)

            many_in_parallel = False # 非并行处理 (Not parallel processing)
            # forwardsim_x_true 和 forwardsim_y已经是numpy数组 (forwardsim_x_true and forwardsim_y are already numpy arrays)
            forwardsim_x_pred_list = dyn_model.do_forward_sim(forwardsim_x_true, forwardsim_y, many_in_parallel) # 执行前向模拟 (Perform forward simulation)
            forwardsim_x_pred = np.array(forwardsim_x_pred_list) # 转换为numpy数组 (Convert to numpy array)

            np.save(save_dir + '/saved_forwardsim/forwardsim_states_true_'+str(counter_agg_iters)+'.npy', forwardsim_x_true) # 保存真实状态 (Save true states)
            np.save(save_dir + '/saved_forwardsim/forwardsim_states_pred_'+str(counter_agg_iters)+'.npy', forwardsim_x_pred) # 保存预测状态 (Save predicted states)

        # ... (MPC控制器执行和数据聚合逻辑与原代码类似，确保与PyTorch兼容) ... (MPC controller execution and data aggregation logic similar to original, ensure PyTorch compatibility)
        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("##############################################") # 打印分隔符 (Print separator)
            print("#### Execute the controller to follow desired trajectories") # 打印执行控制器信息 (Print message about executing controller)
            print("##############################################\n") # 打印分隔符 (Print separator)

        list_rewards=[] # 初始化奖励列表 (Initialize rewards list)
        starting_states_rollouts=[] # 初始化起始状态列表 (Initialize starting states list for rollouts)
        selected_multiple_u_rollouts = [] # 初始化选择的多个u列表 (Initialize list for selected multiple u for rollouts)
        resulting_multiple_x_rollouts = [] # 初始化产生的多个x列表 (Initialize list for resulting multiple x for rollouts)

        horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, desired_snake_headingInit = get_trajfollow_params(which_agent, args.desired_traj_type) # 获取轨迹跟随参数 (Get trajectory following parameters)
        if(follow_trajectories==False): # 如果不跟随轨迹 (If not following trajectories)
            desired_snake_headingInit=0 # 设置期望初始朝向为0 (Set desired initial heading to 0)

        for rollout_num in range(num_trajectories_for_aggregation): # 对于每个聚合轨迹 (For each aggregation trajectory)
            if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
                print("\nPerforming MPC rollout #", rollout_num) # 打印当前MPC rollout编号 (Print current MPC rollout number)

            if(which_agent==2): # 如果是游泳者智能体 (If it's the Swimmer agent)
                starting_observation, starting_state_np = env.reset(evaluating=True, returnStartState=True, isSwimmer=True) # 重置环境 (Reset environment)
            else: # 否则 (Otherwise)
                starting_observation, starting_state_np = env.reset(evaluating=True, returnStartState=True) # 重置环境 (Reset environment)
            if(which_agent==2): # 如果是游泳者智能体 (If it's the Swimmer agent)
                starting_state_np[2] = desired_snake_headingInit # 设置初始朝向 (Set initial heading)
                starting_observation, starting_state_np = env.reset(starting_state_np, returnStartState=True) # 再次重置环境 (Reset environment again)
            
            starting_observation_NNinput_np = from_observation_to_usablestate(starting_observation, which_agent, True) # 转换初始观测 (Convert initial observation)
            desired_x_np = make_trajectory(args.desired_traj_type, starting_observation_NNinput_np, x_index, y_index, which_agent) # 创建期望轨迹 (Create desired trajectory)

            if(noise_actions_during_MPC_rollouts): # 如果MPC rollout期间对动作添加噪声 (If adding noise to actions during MPC rollouts)
                curr_noise_amount = 0.005 # 设置当前噪声量 (Set current noise amount)
            else: # 否则 (Otherwise)
                curr_noise_amount=0 # 设置当前噪声量为0 (Set current noise amount to 0)
            
            # MPC控制器执行rollout (MPC controller performs rollout)
            # 注意：MPCController的perform_rollout可能需要调整以处理PyTorch张量 (Note: MPCController's perform_rollout might need adjustment for PyTorch tensors)
            resulting_x_np, selected_u_np, ep_rew, _ = mpc_controller.perform_rollout(
                                                            starting_state_np, starting_observation,
                                                            starting_observation_NNinput_np, desired_x_np,
                                                            follow_trajectories, horiz_penalty_factor,
                                                            forward_encouragement_factor, heading_penalty_factor,
                                                            noise_actions_during_MPC_rollouts, curr_noise_amount) # 执行rollout (Perform rollout)

            list_rewards.append(ep_rew) # 添加奖励 (Add reward)
            selected_multiple_u_rollouts.append(selected_u_np) # 添加选择的u (Add selected u)
            resulting_multiple_x_rollouts.append(resulting_x_np) # 添加产生的x (Add resulting x)
            starting_states_rollouts.append(starting_state_np) # 添加起始状态 (Add starting state)

        if(args.visualize_MPC_rollout): # 如果可视化MPC rollout (If visualizing MPC rollout)
            input("\n\nPAUSE BEFORE VISUALIZATION... Press Enter to continue...") # 暂停等待用户输入 (Pause and wait for user input)
            for vis_index in range(num_trajectories_for_aggregation): # 对于每个聚合轨迹 (For each aggregation trajectory)
                visualize_rendering(starting_states_rollouts[vis_index], selected_multiple_u_rollouts[vis_index], env, dt_steps, dt_from_xml, which_agent) # 可视化渲染 (Visualize rendering)

        avg_rew = np.mean(np.array(list_rewards)) # 计算平均奖励 (Calculate average reward)
        std_rew = np.std(np.array(list_rewards)) # 计算奖励标准差 (Calculate standard deviation of rewards)
        print("############# Avg reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", avg_rew) # 打印平均奖励 (Print average reward)
        print("############# Std reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", std_rew) # 打印奖励标准差 (Print standard deviation of rewards)
        print("############# Rewards for the ", num_trajectories_for_aggregation, " MPC rollouts: ", list_rewards) # 打印奖励列表 (Print list of rewards)

        list_num_datapoints.append(inputs_normalized.shape[0] + (inputs_new_normalized.shape[0] if dataX_new.shape[0] > 0 else 0) ) # 添加数据点数量 (Add number of data points)
        list_avg_rew.append(avg_rew) # 添加平均奖励 (Add average reward)

        # ... (数据聚合和保存逻辑与原代码类似) ... (Data aggregation and saving logic similar to original code)
        if(counter_agg_iters < (num_aggregation_iters - 1)): # 如果当前聚合迭代次数小于总迭代次数减1 (If current aggregation iteration is less than total iterations minus 1)
            x_array_agg = np.array(resulting_multiple_x_rollouts)[0:(rollouts_forTraining)] # 获取用于训练的x数组 (Get x array for training)
            u_array_agg = np.array(selected_multiple_u_rollouts)[0:(rollouts_forTraining)] # 获取用于训练的u数组 (Get u array for training)

            for i in range(rollouts_forTraining): # 对于每个用于训练的rollout (For each rollout for training)
                x_rollout = np.array(x_array_agg[i]) # 获取当前rollout的x (Get x for current rollout)
                u_rollout = np.squeeze(np.array(u_array_agg[i])) # 获取当前rollout的u并压缩维度 (Get u for current rollout and squeeze dimensions)

                newDataX_iter = np.copy(x_rollout[0:-1, :]) # 新的dataX (New dataX)
                newDataY_iter = np.copy(u_rollout) # 新的dataY (New dataY)
                newDataZ_iter = np.copy(x_rollout[1:, :] - x_rollout[0:-1, :]) # 新的dataZ (New dataZ)

                if(make_aggregated_dataset_noisy): # 如果对聚合数据集添加噪声 (If adding noise to aggregated dataset)
                    newDataX_iter = add_noise(newDataX_iter, noiseToSignal) # 对新的dataX添加噪声 (Add noise to new dataX)
                    newDataZ_iter = add_noise(newDataZ_iter, noiseToSignal) # 对新的dataZ添加噪声 (Add noise to new dataZ)

                dataX_new = np.concatenate((dataX_new, newDataX_iter)) if dataX_new.shape[0] > 0 else newDataX_iter # 连接新的dataX (Concatenate new dataX)
                dataY_new = np.concatenate((dataY_new, newDataY_iter)) if dataY_new.shape[0] > 0 else newDataY_iter # 连接新的dataY (Concatenate new dataY)
                dataZ_new = np.concatenate((dataZ_new, newDataZ_iter)) if dataZ_new.shape[0] > 0 else newDataZ_iter # 连接新的dataZ (Concatenate new dataZ)

            # 更新验证集 (Update validation set)
            x_array_val_agg = np.array(resulting_multiple_x_rollouts)[rollouts_forTraining:len(resulting_multiple_x_rollouts)] # 获取用于验证的x数组 (Get x array for validation)
            u_array_val_agg = np.array(selected_multiple_u_rollouts)[rollouts_forTraining:len(resulting_multiple_x_rollouts)] # 获取用于验证的u数组 (Get u array for validation)

            temp_states_val_list = [sv for sv in states_val] # 临时验证状态列表 (Temporary list for validation states)
            temp_controls_val_list = [cv for cv in controls_val] # 临时验证控制列表 (Temporary list for validation controls)

            for i in range(x_array_val_agg.shape[0]): # 对于每个用于验证的rollout (For each rollout for validation)
                temp_states_val_list.append(np.array(x_array_val_agg[i])[0:-1,:]) # 添加状态到列表 (Add states to list)
                temp_controls_val_list.append(np.squeeze(np.array(u_array_val_agg[i]))) # 添加控制到列表 (Add controls to list)
            states_val = np.array(temp_states_val_list, dtype=object) # 更新验证状态 (Update validation states, use dtype=object for ragged arrays)
            controls_val = np.array(temp_controls_val_list, dtype=object) # 更新验证控制 (Update validation controls, use dtype=object for ragged arrays)


        np.save(save_dir + '/saved_trajfollow/startingstate_iter' + str(counter_agg_iters) +'.npy', starting_states_rollouts[0] if starting_states_rollouts else np.array([])) # 保存起始状态 (Save starting state)
        np.save(save_dir + '/saved_trajfollow/control_iter' + str(counter_agg_iters) +'.npy', selected_multiple_u_rollouts[0] if selected_multiple_u_rollouts else np.array([])) # 保存控制 (Save control)
        np.save(save_dir + '/saved_trajfollow/true_iter' + str(counter_agg_iters) +'.npy', desired_x_np if 'desired_x_np' in locals() else np.array([])) # 保存真实轨迹 (Save true trajectory)
        np.save(save_dir + '/saved_trajfollow/pred_iter' + str(counter_agg_iters) +'.npy', np.array(resulting_multiple_x_rollouts[0] if resulting_multiple_x_rollouts else [])) # 保存预测轨迹 (Save predicted trajectory)

        if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("\n\nDONE WITH BIG LOOP ITERATION ", counter_agg_iters ,"\n\n") # 打印大循环迭代完成信息 (Print message that big loop iteration is done)
            current_total_data = inputs_normalized.shape[0] + (inputs_new_normalized.shape[0] if dataX_new.shape[0] > 0 else 0) # 当前总数据量 (Current total data)
            print("training dataset size: ", current_total_data) # 打印训练数据集大小 (Print training dataset size)
            # print("validation dataset size: ", np.concatenate(states_val).shape[0] if states_val.size > 0 else 0) # 打印验证数据集大小 (Print validation dataset size)
            print("Time taken: {:0.2f} s\n\n".format(time.time()-starting_big_loop)) # 打印耗时 (Print time taken)
        counter_agg_iters= counter_agg_iters+1 # 聚合迭代计数器加1 (Increment aggregation iteration counter)

        np.save(save_dir + '/errors_1_per_agg.npy', np.array(errors_1_per_agg)) # 保存1步误差 (Save 1-step errors)
        np.save(save_dir + '/errors_5_per_agg.npy', np.array(errors_5_per_agg)) # 保存5步误差 (Save 5-step errors)
        np.save(save_dir + '/errors_10_per_agg.npy', np.array(errors_10_per_agg)) # 保存10步误差 (Save 10-step errors)
        np.save(save_dir + '/errors_50_per_agg.npy', np.array(errors_50_per_agg)) # 保存50步误差 (Save 50-step errors)
        np.save(save_dir + '/errors_100_per_agg.npy', np.array(errors_100_per_agg)) # 保存100步误差 (Save 100-step errors)
        np.save(save_dir + '/avg_rollout_rewards_per_agg.npy', np.array(list_avg_rew)) # 保存平均rollout奖励 (Save average rollout rewards)
        np.save(save_dir + '/losses/list_training_loss.npy', np.array(training_loss_list)) # 保存训练损失列表 (Save training loss list)
        np.save(save_dir + '/losses/list_old_loss.npy', np.array(old_loss_list)) # 保存旧损失列表 (Save old loss list)
        np.save(save_dir + '/losses/list_new_loss.npy', np.array(new_loss_list)) # 保存新损失列表 (Save new loss list)

    # ... (为MBMF TRPO使用保存MPC rollouts的逻辑与原代码类似) ... (Logic for saving MPC rollouts for MBMF TRPO usage similar to original code)
    all_rollouts_to_save = [] # 初始化所有要保存的rollout列表 (Initialize list for all rollouts to save)
    if(args.num_rollouts_save_for_mf > 0): # 如果要保存的rollout数量大于0 (If number of rollouts to save is greater than 0)
        print("##############################################") # 打印分隔符 (Print separator)
        print("#### Performing MPC rollouts to save for later mbmf TRPO usage") # 打印执行MPC rollouts信息 (Print message about performing MPC rollouts)
        print("##############################################\n") # 打印分隔符 (Print separator)

        list_rewards_mf = [] # 初始化MF奖励列表 (Initialize MF rewards list)
        starting_states_mf = [] # 初始化MF起始状态列表 (Initialize MF starting states list)
        num_saved = 0 # 初始化已保存数量 (Initialize saved count)
        rollout_num_mf = 0 # 初始化MF rollout编号 (Initialize MF rollout number)
        while(num_saved < args.num_rollouts_save_for_mf): # 当已保存数量小于要保存的数量时 (While saved count is less than number to save)
            if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
                print("\nSo far, saved ", num_saved, " rollouts") # 打印已保存rollout数量 (Print number of saved rollouts)
                print("Currently, on rollout #", rollout_num_mf) # 打印当前rollout编号 (Print current rollout number)

            if(which_agent==2): # 如果是游泳者智能体 (If it's the Swimmer agent)
                starting_observation_mf, starting_state_mf_np = env.reset(evaluating=True, returnStartState=True, isSwimmer=True) # 重置环境 (Reset environment)
            else: # 否则 (Otherwise)
                starting_observation_mf, starting_state_mf_np = env.reset(evaluating=True, returnStartState=True) # 重置环境 (Reset environment)
            if(which_agent==2): # 如果是游泳者智能体 (If it's the Swimmer agent)
                starting_state_mf_np[2] = desired_snake_headingInit # 设置初始朝向 (Set initial heading)
                starting_observation_mf, starting_state_mf_np = env.reset(starting_state_mf_np, returnStartState=True) # 再次重置环境 (Reset environment again)
            starting_observation_NNinput_mf_np = from_observation_to_usablestate(starting_observation_mf, which_agent, True) # 转换初始观测 (Convert initial observation)

            # desired_x_mf_np的定义可能需要，取决于perform_rollout是否总是需要它 (Definition of desired_x_mf_np might be needed if perform_rollout always expects it)
            desired_x_mf_np = make_trajectory(args.desired_traj_type, starting_observation_NNinput_mf_np, x_index, y_index, which_agent) # 创建期望轨迹 (Create desired trajectory)


            startrollout_mf = time.time() # 记录MF rollout开始时间 (Record MF rollout start time)
            curr_noise_amount_mf=0 # 设置当前MF噪声量 (Set current MF noise amount)
            _, _, ep_rew_mf, rollout_saved_mf = mpc_controller.perform_rollout(
                                                                starting_state_mf_np, starting_observation_mf,
                                                                starting_observation_NNinput_mf_np, desired_x_mf_np,
                                                                follow_trajectories, horiz_penalty_factor,
                                                                forward_encouragement_factor, heading_penalty_factor,
                                                                noise_actions_during_MPC_rollouts, curr_noise_amount_mf) # 执行rollout (Perform rollout)

            if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
                print("Time taken for a single rollout: {:0.2f} s\n\n".format(time.time()-startrollout_mf)) # 打印单个rollout耗时 (Print time taken for a single rollout)

            rollout_num_mf += 1 # MF rollout编号加1 (Increment MF rollout number)
            if(ep_rew_mf > min_rew_for_saving): # 如果奖励大于最小保存奖励 (If reward is greater than minimum save reward)
                list_rewards_mf.append(ep_rew_mf) # 添加MF奖励 (Add MF reward)
                all_rollouts_to_save.append(rollout_saved_mf) # 添加要保存的rollout (Add rollout to save)
                starting_states_mf.append(starting_state_mf_np) # 添加MF起始状态 (Add MF starting state)
                num_saved += 1 # 已保存数量加1 (Increment saved count)

        if(len(list_rewards_mf) > 0): # 如果MF奖励列表不为空 (If MF rewards list is not empty)
            avg_rew_mf = np.mean(np.array(list_rewards_mf)) # 计算MF平均奖励 (Calculate MF average reward)
            print("############# Avg over all selected runs (for MF): ", avg_rew_mf) # 打印MF平均奖励 (Print MF average reward)
            print("############# Rewards of all selected runs (for MF): ", list_rewards_mf) # 打印MF奖励列表 (Print MF rewards list)

            pathname_savedMPCrollouts = save_dir + '/savedRollouts_avg'+ str(int(avg_rew_mf)) +'.save' # 定义保存路径1 (Define save path 1)
            pathname2_savedMPCrollouts = save_dir + '/savedRollouts.save' # 定义保存路径2 (Define save path 2)
            with open(pathname_savedMPCrollouts, 'wb') as f: # 打开文件1 (Open file 1)
                cPickle.dump(all_rollouts_to_save, f, protocol=cPickle.HIGHEST_PROTOCOL) # 保存rollouts (Save rollouts)
            with open(pathname2_savedMPCrollouts, 'wb') as f: # 打开文件2 (Open file 2)
                cPickle.dump(all_rollouts_to_save, f, protocol=cPickle.HIGHEST_PROTOCOL) # 保存rollouts (Save rollouts)

            with open(save_dir + '/savedRollouts_startingStates.save', 'wb') as f: # 打开起始状态保存文件 (Open starting states save file)
                cPickle.dump(starting_states_mf, f, protocol=cPickle.HIGHEST_PROTOCOL) # 保存MF起始状态 (Save MF starting states)
   
            print("Saved MPC rollouts for later mbmf TRPO usage.") # 打印已保存MPC rollouts信息 (Print message that MPC rollouts are saved)

    np.save(save_dir + '/datapoints_MB.npy', np.array(list_num_datapoints)) # 保存数据点数量 (Save number of data points)
    np.save(save_dir + '/performance_MB.npy', np.array(list_avg_rew)) # 保存平均奖励 (Save average rewards)

    print("ALL DONE.") # 打印全部完成信息 (Print all done message)

    return # 返回 (Return)

if __name__ == '__main__': # 如果是主模块 (If it's the main module)
    main() # 调用主函数 (Call main function)
