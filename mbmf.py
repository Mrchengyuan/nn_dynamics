import numpy as np # 导入numpy库 (Import numpy library)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块 (Import matplotlib's pyplot module)
import math # 导入math模块 (Import math module)
npr = np.random # 将numpy.random赋给npr (Assign numpy.random to npr)
import torch # 导入torch库 (Import torch library)
import torch.nn as nn # 导入torch.nn模块 (Import torch.nn module)
import torch.nn.functional as F # 从PyTorch导入激活函数等 (Import activation functions etc. from PyTorch)
from six.moves import cPickle # 从six.moves导入cPickle (Import cPickle from six.moves)
from collect_samples import CollectSamples # 从collect_samples导入CollectSamples (Import CollectSamples from collect_samples)
from get_true_action import GetTrueAction # 从get_true_action导入GetTrueAction (Import GetTrueAction from get_true_action)
import os # 导入os模块 (Import os module)
import sys # 导入sys模块 (Import sys module)
import copy # 导入copy模块 (Import copy module)
from helper_funcs import create_env # 从helper_funcs导入create_env (Import create_env from helper_funcs)
from helper_funcs import perform_rollouts # 从helper_funcs导入perform_rollouts (Import perform_rollouts from helper_funcs)
from helper_funcs import add_noise # 从helper_funcs导入add_noise (Import add_noise from helper_funcs)
from helper_funcs import visualize_rendering # 从helper_funcs导入visualize_rendering (Import visualize_rendering from helper_funcs)
import argparse # 导入argparse模块 (Import argparse module)
import pickle # 导入pickle模块 (Import pickle module)

# Garage相关导入 (Garage imports)
from garage import wrap_experiment # 导入garage的wrap_experiment (Import wrap_experiment from garage)
from garage.envs import GymnasiumEnv # 导入garage的GymnasiumEnv包装器 (Import GymnasiumEnv wrapper from garage.envs)
from garage.experiment import Snapshotter # 导入garage的Snapshotter (Import Snapshotter from garage.experiment)
from garage.experiment.deterministic import set_seed as garage_set_seed # 导入garage的set_seed并重命名 (Import garage's set_seed and alias it)
from garage.torch.algos import TRPO as GarageTRPO # 导入garage的TRPO算法 (Import TRPO algorithm from garage)
from garage.torch.policies import GaussianMLPPolicy as GarageGaussianMLPPolicy # 导入garage的高斯MLP策略 (Import GaussianMLPPolicy from garage)
from garage.torch.value_functions import GaussianMLPValueFunction # 导入garage的高斯MLP价值函数 (Import GaussianMLPValueFunction from garage)
from garage.trainer import Trainer # 导入garage的Trainer (Import Trainer from garage)
from dynamics_model import Dyn_Model # 从dynamics_model模块导入Dyn_Model类 (PyTorch version) (Import Dyn_Model class (PyTorch version) from dynamics_model module)


class NNPolicyPyTorch(nn.Module): # 定义一个PyTorch神经网络策略类 (Define a PyTorch neural network policy class)
    def __init__(self, inputSize, outputSize, num_fc_layers, depth_fc_layers, dtype=torch.double, weight_decay_dummy=0.001): # 定义构造函数 (Define constructor), weight_decay_dummy用于匹配签名，但L2由优化器处理 (weight_decay_dummy to match signature, but L2 handled by optimizer)
        super(NNPolicyPyTorch, self).__init__() # 调用父类构造函数 (Call parent class constructor)
        self.layers = nn.ModuleList() # 创建一个ModuleList来存储层 (Create a ModuleList to store layers)
        current_dim = inputSize # 设置当前维度为输入大小 (Set current dimension to input size)
        for i in range(num_fc_layers): # 循环创建全连接层 (Loop to create fully connected layers)
            linear_layer = nn.Linear(current_dim, depth_fc_layers).to(dtype=dtype) # 创建线性层 (Create linear layer)
            nn.init.xavier_uniform_(linear_layer.weight) # 应用Xavier均匀初始化 (Apply Xavier uniform initialization)
            nn.init.zeros_(linear_layer.bias) # 初始化偏置为零 (Initialize biases to zero)
            self.layers.append(linear_layer) # 添加线性层 (Add linear layer)
            self.layers.append(nn.Tanh()) # 添加Tanh激活层 (Add Tanh activation layer)
            current_dim = depth_fc_layers # 更新当前维度 (Update current dimension)

        output_layer = nn.Linear(current_dim, outputSize).to(dtype=dtype) # 创建输出层 (Create output layer)
        nn.init.xavier_uniform_(output_layer.weight) # 应用Xavier均匀初始化 (Apply Xavier uniform initialization)
        nn.init.zeros_(output_layer.bias) # 初始化偏置为零 (Initialize biases to zero)
        self.layers.append(output_layer) # 添加输出层 (Add output layer)

    def forward(self, x): # 定义前向传播函数 (Define forward pass function)
        for layer in self.layers: # 遍历所有层 (Iterate through all layers)
            x = layer(x) # 将输入x传递给当前层 (Pass input x through current layer)
        return x # 返回输出 (Return output)

@wrap_experiment # 使用garage的wrap_experiment装饰器 (Decorate with garage's wrap_experiment)
def run_task_garage(ctxt, variant): # 定义运行Garage任务的函数 (Define function to run Garage task)
    garage_set_seed(variant['seed']) # 设置随机种子 (Set random seed)

    env_name_map = {1: "Ant-v4", 2: "Swimmer-v4", 4: "HalfCheetah-v4", 6: "Hopper-v4", 7: "Walker2d-v4"} # 智能体ID到环境名称的映射 (Mapping from agent ID to environment name)
    env_gymnasium, _ = create_env(variant["which_agent"], seed=variant.get("seed", npr.randint(0, 10000)), render_mode=None) # 创建Gymnasium环境 (Create Gymnasium environment)
    env = GymnasiumEnv(env_gymnasium) # 使用GymnasiumEnv包装环境 (Wrap environment with GymnasiumEnv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 定义PyTorch设备 (Define PyTorch device)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): # 如果MPS可用 (If MPS is available)
        device = torch.device("mps") # 使用MPS设备 (Use MPS device)

    policy = GarageGaussianMLPPolicy(env.spec, # 创建Garage高斯MLP策略 (Create Garage GaussianMLPPolicy)
                                     hidden_sizes=(variant["depth_fc_layers"], variant["depth_fc_layers"]), # 隐藏层大小 (Hidden layer sizes)
                                     hidden_nonlinearity=torch.tanh, # 隐藏层激活函数 (Hidden layer activation function)
                                     output_nonlinearity=None, # 输出层激活函数 (Output layer activation function)
                                     init_std=variant["std_on_mlp_policy"]) # 初始标准差 (Initial standard deviation)

    if "policy_state_dict" in variant and variant["policy_state_dict"] is not None: # 如果策略状态字典存在 (If policy state dictionary exists)
        try: # 尝试 (Try)
            policy._module.load_state_dict(variant["policy_state_dict"]) # 加载状态字典到均值网络 (Load state_dict to the mean network)
            print("Loaded state_dict into GarageGaussianMLPPolicy's mean network (_module).") # 打印加载成功信息 (Print success message)
        except RuntimeError as e: # 捕获运行时错误 (Catch runtime error)
            print(f"Error loading state_dict directly into policy._module: {e}. Parameter shapes or names might mismatch.") # 打印错误信息 (Print error message)
    else: # 否则 (Otherwise)
        print("No policy_state_dict found in variant. GarageGaussianMLPPolicy will use its default initialization.") # 打印未找到状态字典信息 (Print message that state_dict not found)

    policy.to(device) # 将策略移至设备 (Move policy to device)

    value_function = GaussianMLPValueFunction(env_spec=env.spec, # 创建高斯MLP价值函数 (Create GaussianMLPValueFunction)
                                          hidden_sizes=(variant["depth_fc_layers"], variant["depth_fc_layers"]), # 隐藏层大小 (Hidden layer sizes)
                                          hidden_nonlinearity=torch.tanh, # 隐藏层激活函数 (Hidden layer activation function)
                                          output_nonlinearity=None).to(device) # 输出层激活函数并移至设备 (Output layer activation function and move to device)

    trainer = Trainer(ctxt) # 创建训练器 (Create Trainer)
    algo = GarageTRPO(env_spec=env.spec, # 创建garage的TRPO算法实例 (Create garage's TRPO algorithm instance)
                      policy=policy, # 策略 (Policy)
                      value_function=value_function, # 价值函数 (Value function)
                      discount=0.995, # 折扣因子 (Discount factor)
                      sampler_batch_size=variant["trpo_batchsize"], # Garage TRPO 使用 sampler_batch_size (Garage TRPO uses sampler_batch_size)
                      max_episode_length=variant["steps_per_rollout"], # 最大回合长度 (Maximum episode length)
                      )

    trainer.setup(algo, env) # 设置训练器 (Setup trainer)
    trainer.train(n_epochs=variant["num_trpo_iters"], batch_size=variant["trpo_batchsize"]) # 开始训练 (Start training)

if __name__ == '__main__': # 如果是主模块 (If it's the main module)
    parser = argparse.ArgumentParser() # 创建ArgumentParser对象 (Create ArgumentParser object)
    parser.add_argument('--save_trpo_run_num', type=int, default='1')
    parser.add_argument('--run_num', type=int, default=1)
    parser.add_argument('--which_agent', type=int, default=1)
    parser.add_argument('--std_on_mlp_policy', type=float, default=0.5)
    args = parser.parse_args() # 解析命令行参数 (Parse command-line arguments)

    save_trpo_run_num = args.save_trpo_run_num
    run_num = args.run_num
    which_agent = args.which_agent
    print_minimal = getattr(args, 'print_minimal', False)
    std_on_mlp_policy = args.std_on_mlp_policy
    use_existing_pretrained_policy = getattr(args, 'use_existing_pretrained_policy', False) # 安全地获取参数 (Safely get parameter)
    visualize_on_policy_rollouts = getattr(args, 'visualize_on_policy_rollouts', False) # 安全地获取参数 (Safely get parameter)
    visualize_mlp_policy = getattr(args, 'visualize_mlp_policy', False) # 安全地获取参数 (Safely get parameter)
    might_render = getattr(args, 'might_render', False) # 安全地获取参数 (Safely get parameter)


    trpo_batchsize = 50000
    if(which_agent==2):
        batchsize = 512; nEpoch = 70; learning_rate = 0.001
        num_agg_iters = 3; num_rollouts_to_agg= 5; num_rollouts_testperformance = 2
        start_using_noised_actions = 0; do_trpo = True
    elif(which_agent==4):
        batchsize = 512; nEpoch = 300; learning_rate = 0.001
        num_agg_iters = 3; num_rollouts_to_agg= 2; num_rollouts_testperformance = 2
        start_using_noised_actions = 10; do_trpo = True
    elif(which_agent==6):
        batchsize = 512; nEpoch = 200; learning_rate = 0.001
        num_agg_iters = 5; num_rollouts_to_agg= 5; num_rollouts_testperformance = 3
        start_using_noised_actions = 50; do_trpo = True; trpo_batchsize = 25000
    elif(which_agent==1):
        batchsize = 512; nEpoch = 200; learning_rate = 0.001
        num_agg_iters = 5; num_rollouts_to_agg= 5; num_rollouts_testperformance = 3
        start_using_noised_actions = 50; do_trpo = True
    else:
        print(f"Agent {which_agent} params not explicitly defined, using defaults or Ant's.")
        batchsize = 512; nEpoch = 200; learning_rate = 0.001
        num_agg_iters = 5; num_rollouts_to_agg= 5; num_rollouts_testperformance = 3
        start_using_noised_actions = 50; do_trpo = True

    with open('run_'+ str(run_num) + '/params.pkl', 'rb') as f:
        param_dict = pickle.load(f)

    N = param_dict['num_control_samples']
    horizon = param_dict['horizon']
    num_fc_layers_old = param_dict['num_fc_layers']
    depth_fc_layers_old = param_dict['depth_fc_layers']
    lr_olddynmodel = param_dict['lr']
    batchsize_olddynmodel = param_dict['batchsize']
    dt_steps = param_dict['dt_steps']
    steps_per_rollout_garage = param_dict['steps_per_episode']
    seed = param_dict['seed']
    torch_dtype_global = torch.double if param_dict.get('torch_datatype', str(torch.double)) == str(torch.double) else torch.float

    f = open('run_'+ str(run_num)+'/savedRollouts.save', 'rb')
    allData = cPickle.load(f)
    f.close()

    env_name_map = {1: "Ant-v4", 2: "Swimmer-v4", 3: "Reacher-v4", 4: "HalfCheetah-v4", 6: "Hopper-v4", 7: "Walker2d-v4", 0: "PointEnv-v0"} # 添加PointEnv映射 (Add PointEnv mapping)
    env_name_for_gt = env_name_map.get(which_agent, "PointEnv-v0") # 获取环境名称 (Get environment name)

    # 为GetTrueAction和渲染创建一次性环境 (Create one-time env for GetTrueAction and rendering if needed)
    # 注意：如果create_env返回的是rllab包装器，可能需要解包 (Note: if create_env returns rllab wrapper, might need unwrap)
    temp_env_for_gt, dt_from_xml = create_env(which_agent, seed=seed, render_mode='human' if might_render else None) # 创建用于GetTrueAction的环境，根据需要设置渲染模式 (Create env for GetTrueAction with render mode if needed)
    # env_for_spec = GymnasiumEnv(temp_env_for_gt) # Garage TRPO的env_spec需要GymnasiumEnv (Garage TRPO's env_spec needs GymnasiumEnv)


    npr.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not print_minimal:
        print(f"Using device: {device}")

    noise_onpol_rollouts=0.005
    plot=False
    print_frequency = 20
    validation_frequency = 50
    num_fc_layers=2
    depth_fc_layers=64
    save_dir = 'run_'+ str(run_num)+'/mbmf'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    allDataArray=[]
    allControlsArray=[]
    for i in range(len(allData)):
        allDataArray.append(allData[i]['observations'])
        allControlsArray.append(allData[i]['actions'])
    training_data_np=np.concatenate(allDataArray)
    labels_np=np.concatenate(allControlsArray)

    if(len(labels_np.shape)==3):
        labels_np=np.squeeze(labels_np)
    print("\n(total) Data size ", training_data_np.shape[0],"\n\n")

    validnum = 10000
    if((which_agent==6)or(which_agent==2)or(which_agent==1)):
        validnum=700
    num = training_data_np.shape[0]-validnum
    validation_x_np = training_data_np[num:num+validnum,:]
    training_data_np = training_data_np[0:num,:]
    validation_z_np = labels_np[num:num+validnum,:]
    labels_np = labels_np[0:num,:]
    print("\nTraining data size ", training_data_np.shape[0])
    print("Validation data size ", validation_x_np.shape[0],"\n")

    if(args.might_render or getattr(args, 'visualize_mlp_policy', False) or getattr(args, 'visualize_on_policy_rollouts', False) ):
        might_render=True
    else:
        might_render=False
    if(might_render):
        # temp_env_for_render, _ = create_env(which_agent, render_mode='human')
        # if temp_env_for_render:
        #    temp_env_for_render.render()
        #    temp_env_for_render.close()
        pass

    datapoints_used_forMB = np.load('run_'+ str(run_num) + '/datapoints_MB.npy')[-1]
    datapoints_used_to_init_imit = training_data_np.shape[0]
    total_datapoints = datapoints_used_forMB + datapoints_used_to_init_imit
    imit_list_num_datapoints = []
    imit_list_avg_rew = []

    inputSize = training_data_np.shape[1]
    outputSize = labels_np.shape[1]

    policy_torch = NNPolicyPyTorch(inputSize, outputSize, num_fc_layers, depth_fc_layers, dtype=torch_dtype_global).to(device)
    optimizer = torch.optim.Adam(policy_torch.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    # 实例化GetTrueAction (Instantiate GetTrueAction)
    g = GetTrueAction() # 创建GetTrueAction实例 (Create GetTrueAction instance)
    # 加载用于GetTrueAction的动力学模型 (Load dynamics model for GetTrueAction)
    # 确保Dyn_Model的参数与之前训练/保存的旧动力学模型一致 (Ensure Dyn_Model params match the old dynamics model that was trained/saved)
    # 注意：这里的mean_x等是从param_dict加载的原始numpy数组，Dyn_Model构造函数会处理它们 (Note: mean_x etc. here are raw numpy arrays from param_dict, Dyn_Model constructor handles them)
    # Dyn_Model的构造函数需要原始的Numpy均值/标准差 (Dyn_Model's constructor expects raw numpy means/stds)
    expert_dyn_model_params = param_dict # 获取专家动力学模型的参数 (Get expert dynamics model parameters)

    # 从param_dict中提取用于Dyn_Model的参数 (Extract parameters for Dyn_Model from param_dict)
    # 这些是用于加载在main.py中训练的动力学模型的参数 (These are parameters for loading the dynamics model trained in main.py)
    # mean_x, mean_y, mean_z, std_x, std_y, std_z 都是numpy数组 (mean_x, mean_y, mean_z, std_x, std_y, std_z are all numpy arrays)
    mean_x_for_gt = expert_dyn_model_params['mean_x']
    mean_y_for_gt = expert_dyn_model_params['mean_y']
    mean_z_for_gt = expert_dyn_model_params['mean_z']
    std_x_for_gt = expert_dyn_model_params['std_x']
    std_y_for_gt = expert_dyn_model_params['std_y']
    std_z_for_gt = expert_dyn_model_params['std_z']

    g.make_model(env_inp=temp_env_for_gt, rundir='run_'+ str(run_num), # 传递环境实例和运行目录 (Pass environment instance and run directory)
                 num_fc_layers=num_fc_layers_old, depth_fc_layers=depth_fc_layers_old, # 传递旧模型的层配置 (Pass layer configuration of old model)
                 which_agent=which_agent, lr=lr_olddynmodel, batchsize=batchsize_olddynmodel, # 传递智能体和旧模型训练参数 (Pass agent and old model training parameters)
                 N=N, horizon=horizon, steps_per_episode=steps_per_rollout_garage, # 传递MPC参数 (Pass MPC parameters)
                 dt_steps=dt_steps, print_minimal=print_minimal, # 传递其他参数 (Pass other parameters)
                 torch_device=device, torch_dtype=torch_dtype_global) # 传递PyTorch设备和数据类型 (Pass PyTorch device and data type)
    print("GetTrueAction model created and loaded.") # 打印GetTrueAction模型已创建并加载 (Print GetTrueAction model created and loaded)


    policy_state_dict_for_garage = None

    if(not(args.use_existing_pretrained_policy)):
        for agg_iter in range(num_agg_iters):
            print("ON AGGREGATION ITERATION ", agg_iter)
            rewards_for_this_iter=[]
            plot_trainingloss_x=[]
            plot_trainingloss_y=[]
            plot_validloss_x=[]
            plot_validloss_y=[]

            for i in range(nEpoch):
                avg_loss=0
                iters_in_batch=0
                num_train_samples = training_data_np.shape[0]
                shuffled_indices = npr.permutation(num_train_samples)

                for batch_start_idx in range(0, num_train_samples, batchsize):
                    batch_end_idx = min(batch_start_idx + batchsize, num_train_samples)
                    batch_indices = shuffled_indices[batch_start_idx:batch_end_idx]

                    inputs_torch = torch.tensor(training_data_np[batch_indices], dtype=torch_dtype_global, device=device)
                    outputs_torch = torch.tensor(labels_np[batch_indices], dtype=torch_dtype_global, device=device)

                    optimizer.zero_grad()
                    predictions = policy_torch(inputs_torch)
                    loss = loss_fn(predictions, outputs_torch)
                    loss.backward()
                    optimizer.step()
                    avg_loss += math.sqrt(loss.item())
                    iters_in_batch += 1

                current_loss = avg_loss / iters_in_batch if iters_in_batch > 0 else 0
                if(not(print_minimal)):
                    if(i%print_frequency==0):
                        print("training loss: ", current_loss, ", 	nEpoch: ", i)
                plot_trainingloss_x.append(i)
                plot_trainingloss_y.append(current_loss)
                np.save(save_dir + '/plot_trainingloss_x.npy', np.array(plot_trainingloss_x))
                np.save(save_dir + '/plot_trainingloss_y.npy', np.array(plot_trainingloss_y))

                if((i%validation_frequency)==0):
                    avg_valid_loss=0
                    iters_in_valid=0
                    num_val_samples = validation_x_np.shape[0]
                    val_shuffled_indices = npr.permutation(num_val_samples)
                    policy_torch.eval()
                    with torch.no_grad():
                        for val_batch_start_idx in range(0, num_val_samples, batchsize):
                            val_batch_end_idx = min(val_batch_start_idx + batchsize, num_val_samples)
                            val_batch_indices = val_shuffled_indices[val_batch_start_idx:val_batch_end_idx]
                            val_inputs_torch = torch.tensor(validation_x_np[val_batch_indices], dtype=torch_dtype_global, device=device)
                            val_outputs_torch = torch.tensor(validation_z_np[val_batch_indices], dtype=torch_dtype_global, device=device)
                            val_predictions = policy_torch(val_inputs_torch)
                            val_loss = loss_fn(val_predictions, val_outputs_torch)
                            avg_valid_loss += math.sqrt(val_loss.item())
                            iters_in_valid +=1
                    policy_torch.train()
                    curr_valid_loss = avg_valid_loss / iters_in_valid if iters_in_valid > 0 else 0
                    plot_validloss_x.append(i)
                    plot_validloss_y.append(curr_valid_loss)
                    if(not(print_minimal)):
                        print("validation loss: ", curr_valid_loss, ", 	nEpoch: ", i, "\n")
                    np.save(save_dir + '/plot_validloss_x.npy', np.array(plot_validloss_x))
                    np.save(save_dir + '/plot_validloss_y.npy', np.array(plot_validloss_y))

            print("DONE TRAINING NNPolicyPyTorch.")
            print("final training loss: ", current_loss, ", 	nEpoch: ", i)
            print("final validation loss: ", curr_valid_loss, ", 	nEpoch: ", i)

            # DAgger逻辑 (DAgger logic)
            if g is not None: # 如果GetTrueAction对象已创建 (If GetTrueAction object is created)
                print("\n\nCollecting on-policy rollouts for DAgger...\n\n") # 打印收集DAgger的on-policy rollouts信息 (Print message for collecting on-policy rollouts for DAgger)
                starting_states_dagger = [] # 初始化DAgger的初始状态列表 (Initialize list for DAgger starting states)
                observations_dagger = [] # 初始化DAgger的观测列表 (Initialize list for DAgger observations)
                actions_dagger_imit = [] # 初始化DAgger的模仿动作列表 (Initialize list for DAgger imitation actions)
                true_actions_dagger = [] # 初始化DAgger的真实动作列表 (Initialize list for DAgger true actions)

                for rollout_idx in range(num_rollouts_to_agg): # 对于每个要聚合的rollout (For each rollout to aggregate)
                    if(not(print_minimal)): # 如果不是最小化打印 (If not minimal print)
                        print("\nOn DAgger rollout #", rollout_idx) # 打印当前DAgger rollout编号 (Print current DAgger rollout number)

                    obs_dagger_np, info_dagger = temp_env_for_gt.reset() # 重置环境 (Reset environment)
                    # starting_state_dagger = obs_dagger_np # 或者更完整的状态，如果可用 (Or more complete state if available)

                    observations_for_rollout_dagger = [] # 初始化单个rollout的观测列表 (Initialize list for observations for a single rollout)
                    actions_for_rollout_dagger_imit = [] # 初始化单个rollout的模仿动作列表 (Initialize list for imitation actions for a single rollout)
                    true_actions_for_rollout_dagger = [] # 初始化单个rollout的真实动作列表 (Initialize list for true actions for a single rollout)

                    total_rew_dagger = 0 # 初始化DAgger的总奖励 (Initialize total DAgger reward)

                    for step_dagger in range(steps_per_rollout_garage): # 对于每个DAgger步骤 (For each DAgger step)
                        obs_dagger_torch = torch.tensor(obs_dagger_np, dtype=torch_dtype_global, device=device).unsqueeze(0) # 转换为PyTorch张量并添加批次维度 (Convert to PyTorch tensor and add batch dimension)
                        action_imit_torch = policy_torch(obs_dagger_torch) # 获取模仿策略的动作 (Get action from imitation policy)
                        action_imit_np = action_imit_torch.squeeze(0).cpu().detach().numpy() # 转换为Numpy数组 (Convert to NumPy array)

                        if(agg_iter > start_using_noised_actions): # 如果聚合迭代次数超过阈值 (If aggregation iteration exceeds threshold)
                            action_imit_np = action_imit_np + noise_onpol_rollouts*npr.normal(size=action_imit_np.shape) # 添加探索噪声 (Add exploration noise)
                            action_imit_np = np.clip(action_imit_np, env_for_spec.action_space.low, env_for_spec.action_space.high) # 裁剪动作 (Clip action)

                        observations_for_rollout_dagger.append(obs_dagger_np) # 添加观测 (Add observation)
                        actions_for_rollout_dagger_imit.append(action_imit_np) # 添加模仿动作 (Add imitation action)

                        true_action_np = g.get_action(obs_dagger_np) # 从专家获取真实动作 (Get true action from expert)
                        true_actions_for_rollout_dagger.append(true_action_np) # 添加真实动作 (Add true action)

                        next_obs_dagger_np, rew_dagger, terminated_dagger, truncated_dagger, _ = temp_env_for_gt.step(action_imit_np) # 执行一步环境交互 (Perform one step of environment interaction)
                        done_dagger = terminated_dagger or truncated_dagger # 计算完成标志 (Calculate done flag)
                        total_rew_dagger += rew_dagger # 累加奖励 (Accumulate reward)
                        obs_dagger_np = next_obs_dagger_np # 更新观测 (Update observation)

                        if done_dagger: # 如果完成 (If done)
                            break # 跳出循环 (Break loop)

                    # starting_states_dagger.append(starting_state_dagger) # 保存初始状态 (Save initial state)
                    observations_dagger.extend(observations_for_rollout_dagger) # 扩展观测列表 (Extend observations list)
                    # actions_dagger_imit.extend(actions_for_rollout_dagger_imit) # 扩展模仿动作列表 (Extend imitation actions list)
                    labels_np = np.concatenate([labels_np, np.array(true_actions_for_rollout_dagger)], axis=0) if labels_np.size > 0 else np.array(true_actions_for_rollout_dagger) # 连接真实动作到标签 (Concatenate true actions to labels)
                    training_data_np = np.concatenate([training_data_np, np.array(observations_for_rollout_dagger)], axis=0) if training_data_np.size > 0 else np.array(observations_for_rollout_dagger) # 连接观测到训练数据 (Concatenate observations to training data)

                    print(f"DAgger rollout {rollout_idx} reward: {total_rew_dagger}") # 打印DAgger rollout奖励 (Print DAgger rollout reward)

                # 更新用于模仿学习的数据集 (Update dataset for imitation learning)
                # labels_np 和 training_data_np 已在循环中更新 (labels_np and training_data_np are already updated in the loop)
                print(f"DAgger: New training data size: {training_data_np.shape[0]}") # 打印新训练数据大小 (Print new training data size)


        torch.save(policy_torch.state_dict(), save_dir + '/policy_torch_statedict.pth')
        if(not(print_minimal)):
            print(f"Saved PyTorch NNPolicy parameters to {save_dir}/policy_torch_statedict.pth")
        policy_state_dict_for_garage = policy_torch.state_dict()

    else: # use_existing_pretrained_policy is True (如果使用已有的预训练策略)
        policy_path = save_dir + '/policy_torch_statedict.pth' # 定义策略路径 (Define policy path)
        if os.path.exists(policy_path): # 如果路径存在 (If path exists)
            print(f"Loading existing PyTorch policy from {policy_path}") # 打印加载信息 (Print loading message)
            policy_state_dict_for_garage = torch.load(policy_path, map_location=device) # 加载状态字典 (Load state_dict)
        else: # 否则 (Otherwise)
            print(f"ERROR: Pretrained policy file not found at {policy_path}. Exiting.") # 打印错误信息 (Print error message)
            sys.exit() # 退出 (Exit)

    if(do_trpo): # 如果执行TRPO (If executing TRPO)
        variant_dict_garage = dict( # 为Garage TRPO创建variant字典 (Create variant dictionary for Garage TRPO)
            policy_state_dict=policy_state_dict_for_garage, # 传递加载的PyTorch策略状态字典 (Pass loaded PyTorch policy state_dict)
            which_agent=which_agent, # 智能体类型 (Agent type)
            trpo_batchsize=trpo_batchsize, # TRPO批处理大小 (TRPO batch size)
            steps_per_rollout=steps_per_rollout_garage, # Garage中的max_episode_length (max_episode_length in Garage)
            depth_fc_layers=depth_fc_layers, # 全连接层深度 (Depth of FC layers)
            std_on_mlp_policy=std_on_mlp_policy, # MLP策略标准差 (MLP policy standard deviation)
            seed=seed, # 随机种子 (Random seed)
            num_trpo_iters=200 # 示例：为Garage TRPO设置迭代次数 (Example: Set iterations for Garage TRPO) - 根据需要调整 (Adjust as needed)
        )
        exp_name_garage = f'garage_trpo_agent{which_agent}_seed{seed}_run{save_trpo_run_num}' # Garage实验名称 (Garage experiment name)

        run_task_garage(snapshot_config=Snapshotter(snapshot_dir=os.path.join('data/local/experiment', exp_name_garage), snapshot_mode="last", snapshot_gap=1), # 快照配置 (Snapshot configuration)
                        variant=variant_dict_garage, # 变体参数 (Variant parameters)
                        log_dir=os.path.join('data/local/experiment', exp_name_garage), # 日志目录 (Log directory)
                        seed=seed) # 传递种子 (Pass seed)
>>>>>>> REPLACE
