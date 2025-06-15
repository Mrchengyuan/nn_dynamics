import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import numpy.random as npr # 导入numpy的随机数生成模块 (Import numpy's random number generation module)
import torch # 导入torch库 (Import torch library)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import math # 导入math模块，用于数学运算 (Import math module for mathematical operations)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)
from six.moves import cPickle # 从six.moves导入cPickle，用于Python 2/3兼容的pickle (Import cPickle from six.moves for Python 2/3 compatible pickle)
# from feedforward_network import feedforward_network # FeedforwardNetwork现在由Dyn_Model内部使用 (FeedforwardNetwork is now used internally by Dyn_Model)
import os # 导入os模块，用于与操作系统交互 (Import os module for interacting with the operating system)
from data_manipulation import from_observation_to_usablestate # 从data_manipulation模块导入from_observation_to_usablestate函数 (Import from_observation_to_usablestate function from data_manipulation module)
from dynamics_model import Dyn_Model # 从dynamics_model模块导入Dyn_Model类 (PyTorch version) (Import Dyn_Model class (PyTorch version) from dynamics_model module)
from data_manipulation import get_indices # 从data_manipulation模块导入get_indices函数 (Import get_indices function from data_manipulation module)
from mpc_controller import MPCController # 从mpc_controller模块导入MPCController类 (Import MPCController class from mpc_controller module)
from trajectories import make_trajectory # 从trajectories模块导入make_trajectory函数 (Import make_trajectory function from trajectories module)

class GetTrueAction: # 定义GetTrueAction类 (Define GetTrueAction class)

    def make_model(self, env_inp, rundir, num_fc_layers, depth_fc_layers, which_agent,
                    lr, batchsize, N, horizon, steps_per_episode, dt_steps, print_minimal,
                    torch_device, torch_dtype): # 定义创建模型的函数 (Define function to make model)
        
        # 变量 (vars)
        # self.sess = sess # TensorFlow会话已移除 (TensorFlow session removed)
        self.env = copy.deepcopy(env_inp) # 深拷贝环境对象 (Deepcopy the environment object)
        self.N = N  # 控制样本数量 (Number of control samples)
        self.horizon = horizon # 控制时域 (Control horizon)
        self.which_agent = which_agent # 智能体类型 (Agent type)
        self.steps_per_episode = steps_per_episode # 每回合步数 (Steps per episode)
        self.dt_steps = dt_steps # 时间步长 (Time step duration)
        self.print_minimal = print_minimal # 是否最小化打印信息 (Whether to minimize print output)
        self.torch_device = torch_device # PyTorch设备 (PyTorch device)
        self.torch_dtype = torch_dtype # PyTorch数据类型 (PyTorch data type)

        # 获取大小 (get sizes)
        dataX= np.load(rundir + '/training_data/dataX.npy') # 加载dataX (Load dataX)
        dataY= np.load(rundir + '/training_data/dataY.npy') # 加载dataY (Load dataY)
        dataZ= np.load(rundir + '/training_data/dataZ.npy') # 加载dataZ (Load dataZ)
        inputs = np.concatenate((dataX, dataY), axis=1) # 连接输入 (Concatenate inputs)
        assert inputs.shape[0] == dataZ.shape[0] # 确认输入和输出的样本数相同 (Assert that input and output have the same number of samples)
        inputSize = inputs.shape[1] # 输入大小 (Input size)
        outputSize = dataZ.shape[1] # 输出大小 (Output size)

        # 计算均值和标准差 (calculate the means and stds)
        # 这些将作为NumPy数组传递给Dyn_Model，Dyn_Model内部会转换为张量 (These will be passed as NumPy arrays to Dyn_Model, which converts them to tensors internally)
        self.mean_x = np.mean(dataX, axis = 0) # x的均值 (Mean of x)
        # dataX = dataX - self.mean_x # 标准化步骤在Dyn_Model中或数据加载时处理 (Normalization steps handled in Dyn_Model or during data loading)
        self.std_x = np.std(dataX - self.mean_x, axis = 0) # x的标准差 (Standard deviation of x)
        # dataX = np.nan_to_num(dataX/self.std_x)
        self.mean_y = np.mean(dataY, axis = 0)  # y的均值 (Mean of y)
        self.std_y = np.std(dataY - self.mean_y, axis = 0) # y的标准差 (Standard deviation of y)
        self.mean_z = np.mean(dataZ, axis = 0) # z的均值 (Mean of z)
        self.std_z = np.std(dataZ - self.mean_z, axis = 0) # z的标准差 (Standard deviation of z)

        # 获取x和y索引 (get x and y index)
        x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(which_agent) # 获取智能体的索引 (Get indices for the agent)

        # 创建动力学模型 (make dyn model)
        # PyTorch模型初始化不需要sess，tf_datatype (PyTorch model initialization does not require sess, tf_datatype)
        self.dyn_model = Dyn_Model(inputSize, outputSize, lr, batchsize, which_agent, x_index, y_index, num_fc_layers,
                                    depth_fc_layers, self.mean_x, self.mean_y, self.mean_z, self.std_x, self.std_y, self.std_z, 
                                    self.print_minimal, device=self.torch_device) # 创建Dyn_Model实例 (Create Dyn_Model instance)
        self.dyn_model.to(self.torch_dtype) # 将模型参数转换为指定的torch_dtype (Convert model parameters to the specified torch_dtype)

        # PyTorch中不需要显式全局变量初始化 (Explicit global variable initialization is not needed in PyTorch)
        # self.sess.run(tf.global_variables_initializer())

        # 从期望的已训练动力学模型加载权重 (load in weights from desired trained dynamics model)
        pathname = rundir + '/models/finalModel.pth' # PyTorch模型文件通常使用.pth扩展名 (PyTorch model files typically use .pth extension)
        # TensorFlow Saver已移除 (TensorFlow Saver removed)
        # saver = tf.train.Saver(max_to_keep=0)
        # saver.restore(self.sess, pathname)
        if os.path.exists(pathname): # 如果路径存在 (If path exists)
            self.dyn_model.load_weights(pathname) # 加载模型权重 (Load model weights)
            print("\n\nRestored dynamics model with variables from ", pathname,"\n\n") # 打印模型恢复信息 (Print model restoration message)
        else: # 否则 (Otherwise)
            print(f"\n\nWARNING: Dynamics model checkpoint not found at {pathname}. Using randomly initialized model.\n\n") # 打印警告信息 (Print warning message)


        # 创建控制器，用于查询最优动作 (make controller, to use for querying optimal action)
        self.mpc_controller = MPCController(self.env, self.dyn_model, self.horizon, self.which_agent, self.steps_per_episode, 
                                            self.dt_steps, self.N,
                                            self.mean_x.cpu().numpy(), self.mean_y.cpu().numpy(), self.mean_z.cpu().numpy(), # 将PyTorch张量转换为NumPy数组 (Convert PyTorch tensors to NumPy arrays)
                                            self.std_x.cpu().numpy(), self.std_y.cpu().numpy(), self.std_z.cpu().numpy(), # 将PyTorch张量转换为NumPy数组 (Convert PyTorch tensors to NumPy arrays)
                                            'nc', self.print_minimal, x_index, y_index, z_index, yaw_index, joint1_index,
                                            joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index,
                                            torch_device=self.torch_device, torch_dtype=self.torch_dtype) # 创建MPCController实例 (Create MPCController instance)
        # desired_states现在应为PyTorch张量 (desired_states should now be a PyTorch tensor)
        desired_states_np = make_trajectory('straight', np.zeros((100,)), x_index, y_index, which_agent) # 创建轨迹 (Create trajectory)
        self.mpc_controller.desired_states = torch.tensor(desired_states_np, dtype=self.torch_dtype, device=self.torch_device) # 转换为张量 (Convert to tensor)

        # 选择任务或奖励函数 (select task or reward func)
        # 确保RewardFunctions可以处理NumPy或PyTorch张量 (Ensure RewardFunctions can handle NumPy or PyTorch tensors)
        self.reward_func = self.mpc_controller.reward_functions.get_reward_func(False, self.mpc_controller.desired_states.cpu().numpy(), 0, 0, 0) # 获取奖励函数 (Get reward function)


    def get_action(self, curr_obs_np): # 定义获取动作的函数 (Define function to get action)

        curr_nn_state_np = from_observation_to_usablestate(curr_obs_np, self.which_agent, True) # 将观测转换为可用状态 (Convert observation to usable state)
        curr_nn_state_torch = torch.tensor(curr_nn_state_np, dtype=self.torch_dtype, device=self.torch_device) # 转换为PyTorch张量 (Convert to PyTorch tensor)

        # MPCController.get_action现在期望PyTorch张量 (MPCController.get_action now expects PyTorch tensors)
        best_action_np, _, _, _ = self.mpc_controller.get_action(curr_nn_state_torch, 0, self.reward_func) # 获取最佳动作 (Get best action)

        return best_action_np # 返回最佳动作 (Return best action)