import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import numpy.random as npr # 导入numpy的随机数生成模块 (Import numpy's random number generation module)
import torch # 导入torch库 (Import torch library)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import math # 导入math模块，用于数学运算 (Import math module for mathematical operations)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)
from six.moves import cPickle # 从six.moves导入cPickle，用于Python 2/3兼容的pickle (Import cPickle from six.moves for Python 2/3 compatible pickle)
from rllab.misc import tensor_utils # 从rllab.misc导入tensor_utils (Import tensor_utils from rllab.misc) - 注意：此模块可能与PyTorch不直接兼容，后续可能需要检查 (Note: This module might not be directly compatible with PyTorch and may need checking later)
from data_manipulation import from_observation_to_usablestate # 从data_manipulation模块导入from_observation_to_usablestate函数 (Import from_observation_to_usablestate function from data_manipulation module)
from reward_functions import RewardFunctions # 从reward_functions模块导入RewardFunctions类 (Import RewardFunctions class from reward_functions module)

class MPCController: # 定义MPCController类 (Define MPCController class)

    def __init__(self, env_inp, dyn_model, horizon, which_agent, steps_per_episode, dt_steps, num_control_samples, 
                mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index, z_index, yaw_index, 
                joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index,
                torch_device='cpu', torch_dtype=torch.double): # 定义构造函数，添加torch_device和torch_dtype参数 (Define constructor, add torch_device and torch_dtype parameters)

        # 初始化变量 (init vars)
        self.env = copy.deepcopy(env_inp) # 深拷贝环境对象 (Deepcopy the environment object)
        self.N = num_control_samples # 控制样本数量 (Number of control samples)
        self.which_agent = which_agent # 智能体类型 (Agent type)
        self.horizon = horizon # 控制时域 (Control horizon)
        self.dyn_model = dyn_model # 动力学模型 (Dynamics model)
        self.steps_per_episode = steps_per_episode  # 每回合步数 (Steps per episode)
        self.torch_device = torch_device # PyTorch设备 (PyTorch device)
        self.torch_dtype = torch_dtype # PyTorch数据类型 (PyTorch data type)

        # 将均值和标准差转换为PyTorch张量并存储 (Convert means and stds to PyTorch tensors and store them)
        self.mean_x = torch.tensor(mean_x, dtype=self.torch_dtype, device=self.torch_device) # x的均值 (Mean of x)
        self.mean_y = torch.tensor(mean_y, dtype=self.torch_dtype, device=self.torch_device) # y的均值 (Mean of y)
        self.mean_z = torch.tensor(mean_z, dtype=self.torch_dtype, device=self.torch_device) # z的均值 (Mean of z)
        self.std_x = torch.tensor(std_x, dtype=self.torch_dtype, device=self.torch_device) # x的标准差 (Standard deviation of x)
        self.std_y = torch.tensor(std_y, dtype=self.torch_dtype, device=self.torch_device) # y的标准差 (Standard deviation of y)
        self.std_z = torch.tensor(std_z, dtype=self.torch_dtype, device=self.torch_device) # z的标准差 (Standard deviation of z)

        self.x_index = x_index # x索引 (x index)
        self.y_index = y_index # y索引 (y index)
        self.z_index = z_index  # z索引 (z index)
        self.yaw_index = yaw_index # 偏航角索引 (Yaw index)
        self.joint1_index = joint1_index # 关节1索引 (Joint 1 index)
        self.joint2_index = joint2_index # 关节 2索引 (Joint 2 index)
        self.frontleg_index = frontleg_index # 前腿索引 (Front leg index)
        self.frontshin_index = frontshin_index # 前胫索引 (Front shin index)
        self.frontfoot_index = frontfoot_index # 前脚索引 (Front foot index)
        self.xvel_index = xvel_index # x速度索引 (x velocity index)
        self.orientation_index = orientation_index # 方向索引 (Orientation index)
        self.actions_ag = actions_ag # 聚合动作类型 (Aggregated actions type)
        self.print_minimal = print_minimal # 是否最小化打印信息 (Whether to minimize print output)
        self.reward_functions = RewardFunctions(self.which_agent, self.x_index, self.y_index, self.z_index, self.yaw_index, 
                                                self.joint1_index, self.joint2_index, self.frontleg_index, self.frontshin_index, 
                                                self.frontfoot_index, self.xvel_index, self.orientation_index) # 初始化奖励函数对象 (Initialize RewardFunctions object)

    def perform_rollout(self, starting_fullenvstate, starting_observation, starting_observation_NNinput, desired_states_np, follow_trajectories,
                        horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, noise_actions, noise_amount): # 定义执行rollout的函数 (Define function to perform rollout)

        # 用于保存信息的列表 (lists for saving info)
        traj_taken=[] # 进入NN的状态列表 (list of states that go into NN)
        actions_taken=[] # 执行的动作列表 (list of actions taken)
        observations = [] # 观测列表（环境的直接输出）(list of observations (direct output of the env))
        rewards = [] # 奖励列表 (list of rewards)
        agent_infos = [] # 智能体信息列表 (list of agent infos)
        env_infos = [] # 环境信息列表 (list of env infos)

        # 初始化变量 (init vars)
        stop_taking_steps = False # 是否停止步进的标志 (Flag to stop taking steps)
        total_reward_for_episode = 0 # 当前回合的总奖励 (Total reward for the current episode)
        step=0 # 当前步数 (Current step)
        curr_line_segment = 0 # 当前线段 (Current line segment)
        self.horiz_penalty_factor = horiz_penalty_factor # 水平惩罚因子 (Horizontal penalty factor)
        self.forward_encouragement_factor = forward_encouragement_factor # 前向鼓励因子 (Forward encouragement factor)
        self.heading_penalty_factor = heading_penalty_factor # 朝向惩罚因子 (Heading penalty factor)

        # 扩展期望状态列表，以防用尽 (extend the list of desired states so you don't run out)
        temp = np.tile(np.expand_dims(desired_states_np[-1], axis=0), (10,1)) # 创建重复的最后一个期望状态 (Create repeated last desired state)
        self.desired_states_np = np.concatenate((desired_states_np, temp)) # 连接期望状态 (Concatenate desired states)
        self.desired_states = torch.tensor(self.desired_states_np, dtype=self.torch_dtype, device=self.torch_device) # 将期望状态转换为PyTorch张量 (Convert desired states to PyTorch tensor)


        # 将环境重置到给定的完整环境状态 (reset env to the given full env state)
        if(self.which_agent==5): # 如果是Humanoid-like agent (If it's Humanoid-like agent)
            self.env.reset() # 重置环境 (Reset environment)
        else: # 否则 (Otherwise)
            self.env.reset(starting_fullenvstate) # 使用给定的起始状态重置环境 (Reset environment with given starting state)

        # 当前观测 (current observation)
        obs_np = np.copy(starting_observation) # 复制起始观测 (Copy starting observation)
        # 当前观测，格式适用于NN (current observation in the right format for NN)
        curr_state_np = np.copy(starting_observation_NNinput) # 复制起始NN输入观测 (Copy starting NN input observation)
        traj_taken.append(curr_state_np) # 添加到轨迹列表 (Add to trajectory list)

        # 选择任务或奖励函数 (select task or reward func)
        # 注意：RewardFunctions可能需要调整以适配PyTorch张量或在内部处理转换 (Note: RewardFunctions might need adjustment for PyTorch tensors or handle conversion internally)
        reward_func = self.reward_functions.get_reward_func(follow_trajectories, self.desired_states_np, horiz_penalty_factor,
                                                            forward_encouragement_factor, heading_penalty_factor) # 获取奖励函数 (Get reward function)

        # 根据选择的任务/奖励函数执行步骤 (take steps according to the chosen task/reward function)
        while(stop_taking_steps==False): # 当未停止步进时 (While not stopped taking steps)

            # 获取最优动作 (get optimal action)
            # 将curr_state_np转换为PyTorch张量 (Convert curr_state_np to PyTorch tensor)
            curr_state_torch = torch.tensor(curr_state_np, dtype=self.torch_dtype, device=self.torch_device) # 当前状态的PyTorch张量 (PyTorch tensor for current state)
            best_action_np, best_sim_number, best_sequence_np, moved_to_next = self.get_action(curr_state_torch, curr_line_segment, reward_func) # 获取动作 (Get action)

            # 推进当前所在的线段 (advance which line segment we are on)
            if(follow_trajectories): # 如果跟随轨迹 (If following trajectories)
                if(moved_to_next[best_sim_number]==1): # 如果移动到下一个线段 (If moved to the next line segment)
                    curr_line_segment+=1 # 线段索引加1 (Increment line segment index)
                    print("MOVED ON TO LINE SEGMENT ", curr_line_segment) # 打印移动到新线段信息 (Print message about moving to new line segment)

            # 对动作添加噪声 (noise the action)
            action_to_take_np = np.copy(best_action_np) # 复制最优动作 (Copy best action)

            # 是否执行带噪声或干净的动作 (whether to execute noisy or clean actions)
            if(self.actions_ag=='nn'): # 如果聚合类型是nn (If aggregation type is nn)
                apply_noise_to_action = True # 应用噪声 (Apply noise)
            elif(self.actions_ag=='nc'): # 如果聚合类型是nc (If aggregation type is nc)
                apply_noise_to_action = True # 应用噪声 (Apply noise)
            elif(self.actions_ag=='cc'): # 如果聚合类型是cc (If aggregation type is cc)
                apply_noise_to_action = False # 不应用噪声 (Do not apply noise)
            else: # 否则 (Otherwise)
                apply_noise_to_action = noise_actions # 使用传入的noise_actions标志 (Use the passed noise_actions flag)


            clean_action_np = np.copy(action_to_take_np) # 复制干净的动作 (Copy clean action)
            if(apply_noise_to_action): # 如果应用噪声 (If applying noise)
                action_noise = noise_amount * npr.normal(size=action_to_take_np.shape) # 生成噪声 (Generate noise)
                action_to_take_np = action_to_take_np + action_noise # 添加噪声 (Add noise)
                action_to_take_np =np.clip(action_to_take_np, self.env.action_space.low, self.env.action_space.high) # 裁剪动作到有效范围 (Clip action to valid range)


            # 执行动作 (execute the action)
            next_obs_np, rew, done, env_info = self.env.step(action_to_take_np, collectingInitialData=False) # 执行一步环境交互 (Perform one step of environment interaction)

            # 检查是否完成 (check if done)
            if(done): # 如果完成 (If done)
                stop_taking_steps=True # 设置停止标志为True (Set stop flag to True)
            else: # 否则 (Otherwise)
                # 保存信息 (save things)
                observations.append(obs_np) # 添加观测 (Add observation)
                rewards.append(rew) # 添加奖励 (Add reward)
                env_infos.append(env_info) # 添加环境信息 (Add environment info)
                total_reward_for_episode += rew # 累加回合总奖励 (Accumulate total episode reward)

                # 是否保存干净或带噪声的动作 (whether to save clean or noisy actions)
                if(self.actions_ag=='nn'): # 如果聚合类型是nn (If aggregation type is nn)
                    actions_taken.append(np.array([action_to_take_np])) # 保存带噪声的动作 (Save noisy action)
                elif(self.actions_ag=='nc'): # 如果聚合类型是nc (If aggregation type is nc)
                    actions_taken.append(np.array([clean_action_np])) # 保存干净的动作 (Save clean action)
                elif(self.actions_ag=='cc'): # 如果聚合类型是cc (If aggregation type is cc)
                    actions_taken.append(np.array([clean_action_np])) # 保存干净的动作 (Save clean action)
                else: # 否则 (Otherwise) # 默认保存执行的动作 (Default to saving the executed action)
                    actions_taken.append(np.array([action_to_take_np])) # 保存执行的动作 (Save executed action)


                # 这是在环境中执行一步后返回的观测 (this is the observation returned by taking a step in the env)
                obs_np=np.copy(next_obs_np) # 复制下一个观测 (Copy next observation)

                # 获取下一个状态（可用于NN）(get the next state (usable by NN))
                just_one=True # 单个状态标志 (Single state flag)
                next_state_np = from_observation_to_usablestate(next_obs_np, self.which_agent, just_one) # 转换下一个状态 (Convert next state)
                curr_state_np =np.copy(next_state_np) # 复制当前状态 (Copy current state)
                traj_taken.append(curr_state_np) # 添加到轨迹列表 (Add to trajectory list)

                # 记录 (bookkeeping)
                if(not(self.print_minimal)): # 如果不是最小化打印 (If not minimal print)
                    if(step%100==0): # 如果步数是100的倍数 (If step is a multiple of 100)
                        print("done step ", step, ", rew: ", total_reward_for_episode) # 打印步数和奖励信息 (Print step and reward information)
                step+=1 # 步数加1 (Increment step)

                # 何时停止 (when to stop)
                if(follow_trajectories): # 如果跟随轨迹 (If following trajectories)
                    if((step>=self.steps_per_episode) or (curr_line_segment>5)): # 如果达到最大步数或线段数 (If maximum steps or line segments reached)
                        stop_taking_steps = True # 设置停止标志为True (Set stop flag to True)
                else: # 否则 (Otherwise)
                    if(step>=self.steps_per_episode): # 如果达到最大步数 (If maximum steps reached)
                        stop_taking_steps = True # 设置停止标志为True (Set stop flag to True)

        if(not(self.print_minimal)): # 如果不是最小化打印 (If not minimal print)
            print("DONE TAKING ", step, " STEPS.") # 打印完成步数信息 (Print message about completed steps)
            print("Reward: ", total_reward_for_episode) # 打印总奖励 (Print total reward)

        # tensor_utils.stack_tensor_list 和 stack_tensor_dict_list 来自rllab，它们处理numpy数组列表 (tensor_utils.stack_tensor_list and stack_tensor_dict_list are from rllab, they handle lists of numpy arrays)
        # 如果这些函数不直接处理PyTorch张量，确保输入是NumPy数组 (If these functions don't handle PyTorch tensors directly, ensure inputs are NumPy arrays)
        mydict = dict( # 创建字典 (Create dictionary)
        observations=tensor_utils.stack_tensor_list(observations), # 观测 (Observations)
        actions=tensor_utils.stack_tensor_list(actions_taken), # 动作 (Actions)
        rewards=tensor_utils.stack_tensor_list(rewards), # 奖励 (Rewards)
        agent_infos=agent_infos, # 智能体信息 (Agent infos)
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos)) # 环境信息 (Environment infos)

        return traj_taken, actions_taken, total_reward_for_episode, mydict # 返回轨迹、动作、总奖励和字典 (Return trajectory, actions, total reward, and dictionary)

    def get_action(self, curr_nn_state_torch, curr_line_segment, reward_func): # 定义获取动作的函数 (Define function to get action)
        # 随机采样N个候选动作序列 (randomly sample N candidate action sequences)
        all_samples_np = npr.uniform(self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0])) # 生成随机样本 (Generate random samples)
        all_samples_torch = torch.tensor(all_samples_np, dtype=self.torch_dtype, device=self.torch_device) # 将样本转换为PyTorch张量 (Convert samples to PyTorch tensor)

        # 并行地对动作序列进行前向模拟，以获得（预测的）结果轨迹 (forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories)
        many_in_parallel = True # 并行处理标志 (Flag for parallel processing)
        # dyn_model.do_forward_sim 现在需要PyTorch张量 (dyn_model.do_forward_sim now expects PyTorch tensors)
        # curr_nn_state_torch 已经是张量 (curr_nn_state_torch is already a tensor)
        resulting_states_list = self.dyn_model.do_forward_sim(curr_nn_state_torch, all_samples_torch, many_in_parallel) # 执行前向模拟 (Perform forward simulation)

        # resulting_states_list 是一个Python列表，包含Numpy数组 (resulting_states_list is a Python list of Numpy arrays)
        # 我们需要将其转换为一个大的Numpy数组，然后是PyTorch张量，或者直接处理列表中的张量 (We need to convert it to a large Numpy array then PyTorch tensor, or handle tensors in the list directly)
        # 假设 dyn_model.do_forward_sim 返回的是一个Numpy数组列表，每个元素是 (state_dim,)
        # 或者它可能返回一个PyTorch张量列表。根据 dynamics_model.py 的实现，它返回numpy数组列表。
        # 我们需要堆叠它们以形成 [horizon+1, N, statesize]
        resulting_states_np = np.array(resulting_states_list) # 转换为numpy数组 (Convert to numpy array) # this is [horizon+1, N, statesize]
        resulting_states_torch = torch.tensor(resulting_states_np, dtype=self.torch_dtype, device=self.torch_device) # 转换为PyTorch张量 (Convert to PyTorch tensor)


        # 初始化用于评估轨迹的变量 (init vars to evaluate the trajectories)
        scores_np=np.zeros((self.N,)) # 分数numpy数组 (Scores numpy array)
        done_forever_np=np.zeros((self.N,)) # 完成标志numpy数组 (Done flag numpy array)
        # move_to_next_np=np.zeros((self.N,)) # 移动到下一个标志numpy数组 (Move to next flag numpy array) - 在循环中更新 (Updated in loop)
        curr_seg_np = np.tile(curr_line_segment,(self.N,)) # 当前线段numpy数组 (Current segment numpy array)
        curr_seg_np = curr_seg_np.astype(int) # 转换为整数类型 (Convert to integer type)
        prev_forward_np = np.zeros((self.N,)) # 先前的前向numpy数组 (Previous forward numpy array)
        moved_to_next_np = np.zeros((self.N,)) # 移动到下一个标志numpy数组 (Move to next flag numpy array)
        prev_pt_torch = resulting_states_torch[0] # 上一个点的PyTorch张量 (PyTorch tensor for previous point)

        # 累积每个时间步的奖励 (accumulate reward over each timestep)
        for pt_number in range(resulting_states_torch.shape[0]): # 对于每个时间点 (For each time point)

            # “点”的数组...用于每个模拟 (array of "the point"... for each sim)
            pt_torch = resulting_states_torch[pt_number] # N x state, 当前点的PyTorch张量 (PyTorch tensor for current point)

            # 点距离期望轨迹多远 (how far is the point from the desired trajectory)
            # 自上一个点以来，您沿着期望轨迹移动了多远 (how far along the desired traj have you moved since the last point)
            # 注意：calculate_geometric_trajfollow_quantities 需要调整以使用PyTorch张量或接受NumPy并返回NumPy (Note: calculate_geometric_trajfollow_quantities needs adjustment for PyTorch tensors or to accept NumPy and return NumPy)
            min_perp_dist_np, curr_forward_np, curr_seg_np, moved_to_next_np = self.calculate_geometric_trajfollow_quantities(
                                                                                    pt_torch.cpu().numpy(), # 转换为NumPy (Convert to NumPy)
                                                                                    curr_seg_np,
                                                                                    moved_to_next_np) # 计算几何量 (Calculate geometric quantities)

            # 更新奖励分数 (update reward score)
            # reward_func 也可能需要处理PyTorch张量或NumPy (reward_func might also need to handle PyTorch tensors or NumPy)
            scores_np, done_forever_np = reward_func(pt_torch.cpu().numpy(), prev_pt_torch.cpu().numpy(), scores_np,
                                                min_perp_dist_np, curr_forward_np, prev_forward_np, curr_seg_np,
                                                moved_to_next_np, done_forever_np, all_samples_np, pt_number) # 更新奖励分数 (Update reward scores)

            # 更新变量 (update vars)
            prev_forward_np = np.copy(curr_forward_np) # 复制当前前向 (Copy current forward)
            prev_pt_torch = pt_torch.clone() # 克隆当前点 (Clone current point)

        # 选择最佳动作序列 (pick best action sequence)
        best_score = np.min(scores_np) # 最佳分数 (Best score)
        best_sim_number = np.argmin(scores_np)  # 最佳模拟编号 (Best simulation number)
        best_sequence_np = all_samples_np[best_sim_number] # 最佳序列 (Best sequence)
        best_action_np = np.copy(best_sequence_np[0]) # 最佳动作 (Best action)
        
        return best_action_np, best_sim_number, best_sequence_np, moved_to_next_np # 返回最佳动作、模拟编号、序列和移动标志 (Return best action, simulation number, sequence, and moved flag)

    def calculate_geometric_trajfollow_quantities(self, pt_np, curr_seg_np, moved_to_next_np): # 定义计算几何轨迹跟随量的函数 (Define function to calculate geometric trajectory following quantities)
        # 此函数已使用NumPy，因此输入pt_np应为NumPy数组 (This function already uses NumPy, so input pt_np should be a NumPy array)
        # desired_states 也应为NumPy数组 (desired_states should also be a NumPy array)

        # 线段点的数组...用于每个模拟 (arrays of line segment points... for each sim)
        curr_start_np = self.desired_states_np[curr_seg_np] # 当前起点的NumPy数组 (NumPy array for current start)
        curr_end_np = self.desired_states_np[curr_seg_np+1] # 当前终点的NumPy数组 (NumPy array for current end)
        next_start_np = self.desired_states_np[curr_seg_np+1] # 下一个起点的NumPy数组 (NumPy array for next start)
        next_end_np = self.desired_states_np[curr_seg_np+2] # 下一个终点的NumPy数组 (NumPy array for next end)

        # 初始化 (initialize)
        min_perp_dist_np = np.ones((self.N, ))*5000 # 最小垂直距离NumPy数组 (NumPy array for minimum perpendicular distance)

        ####################################### 点到当前线段的最近距离 (closest distance from point to current line segment)

        # 变量 (vars)
        a = pt_np[:,self.x_index]- curr_start_np[:,0] # a变量 (a variable)
        b = pt_np[:,self.y_index]- curr_start_np[:,1] # b变量 (b variable)
        c = curr_end_np[:,0]- curr_start_np[:,0] # c变量 (c variable)
        d = curr_end_np[:,1]- curr_start_np[:,1] # d变量 (d variable)

        # 将点投影到线段上 (project point onto line segment)
        # 使用 np.divide 并处理分母为零的情况 (Use np.divide and handle division by zero)
        dot_product = np.multiply(a,c) + np.multiply(b,d) # 点积 (Dot product)
        len_sq = np.multiply(c,c) + np.multiply(d,d) # 长度平方 (Length squared)
        which_line_section_np = np.divide(dot_product, len_sq, out=np.zeros_like(dot_product), where=len_sq!=0) # 线段上的哪个部分 (Which part of the line segment)


        # 线段上离点最近的点 (point on line segment that's closest to the pt)
        closest_pt_x_np = np.copy(which_line_section_np) # x坐标的最近点 (Closest point x-coordinate)
        closest_pt_y_np = np.copy(which_line_section_np) # y坐标的最近点 (Closest point y-coordinate)
        closest_pt_x_np[which_line_section_np<0] = curr_start_np[:,0][which_line_section_np<0] # 处理小于0的情况 (Handle case less than 0)
        closest_pt_y_np[which_line_section_np<0] = curr_start_np[:,1][which_line_section_np<0] # 处理小于0的情况 (Handle case less than 0)
        closest_pt_x_np[which_line_section_np>1] = curr_end_np[:,0][which_line_section_np>1] # 处理大于1的情况 (Handle case greater than 1)
        closest_pt_y_np[which_line_section_np>1] = curr_end_np[:,1][which_line_section_np>1] # 处理大于1的情况 (Handle case greater than 1)

        mask_on_segment = np.logical_and(which_line_section_np<=1, which_line_section_np>=0) # 在线段上的掩码 (Mask for on the segment)
        closest_pt_x_np[mask_on_segment] = (curr_start_np[:,0] + np.multiply(which_line_section_np,c))[mask_on_segment] # 计算在线段上的x坐标 (Calculate x-coordinate on the segment)
        closest_pt_y_np[mask_on_segment] = (curr_start_np[:,1] + np.multiply(which_line_section_np,d))[mask_on_segment] # 计算在线段上的y坐标 (Calculate y-coordinate on the segment)


        # 点到该最近点的最小距离（即点到线段的最近距离）(min dist from pt to that closest point (ie closes dist from pt to line segment))
        min_perp_dist_np = np.sqrt((pt_np[:,self.x_index]-closest_pt_x_np)**2 + (pt_np[:,self.y_index]-closest_pt_y_np)**2) # 计算最小垂直距离 (Calculate minimum perpendicular distance)


        ####################################### 点的“前向性”...对于每个模拟 ("forward-ness" of the pt... for each sim)
        curr_forward_np = which_line_section_np # 当前前向 (Current forward)

        ###################################### 点到下一线段的最近距离 (closest distance from point to next line segment)

        # 变量 (vars)
        a_next = pt_np[:,self.x_index]- next_start_np[:,0] # a_next变量 (a_next variable)
        b_next = pt_np[:,self.y_index]- next_start_np[:,1] # b_next变量 (b_next variable)
        c_next = next_end_np[:,0]- next_start_np[:,0] # c_next变量 (c_next variable)
        d_next = next_end_np[:,1]- next_start_np[:,1] # d_next变量 (d_next variable)

        # 将点投影到线段上 (project point onto line segment)
        dot_product_next = np.multiply(a_next,c_next) + np.multiply(b_next,d_next) # 下一个点积 (Next dot product)
        len_sq_next = np.multiply(c_next,c_next) + np.multiply(d_next,d_next) # 下一个长度平方 (Next length squared)
        which_line_section_next_np = np.divide(dot_product_next, len_sq_next, out=np.zeros_like(dot_product_next), where=len_sq_next!=0) # 下一线段上的哪个部分 (Which part of the next line segment)


        # 线段上离点最近的点 (point on line segment that's closest to the pt)
        closest_pt_x_next_np = np.copy(which_line_section_next_np) # 下一个x坐标的最近点 (Next closest point x-coordinate)
        closest_pt_y_next_np = np.copy(which_line_section_next_np) # 下一个y坐标的最近点 (Next closest point y-coordinate)
        closest_pt_x_next_np[which_line_section_next_np<0] = next_start_np[:,0][which_line_section_next_np<0] # 处理小于0的情况 (Handle case less than 0)
        closest_pt_y_next_np[which_line_section_next_np<0] = next_start_np[:,1][which_line_section_next_np<0] # 处理小于0的情况 (Handle case less than 0)
        closest_pt_x_next_np[which_line_section_next_np>1] = next_end_np[:,0][which_line_section_next_np>1] # 处理大于1的情况 (Handle case greater than 1)
        closest_pt_y_next_np[which_line_section_next_np>1] = next_end_np[:,1][which_line_section_next_np>1] # 处理大于1的情况 (Handle case greater than 1)
        
        mask_on_next_segment = np.logical_and(which_line_section_next_np<=1, which_line_section_next_np>=0) # 在下一线段上的掩码 (Mask for on the next segment)
        closest_pt_x_next_np[mask_on_next_segment] = (next_start_np[:,0] + np.multiply(which_line_section_next_np,c_next))[mask_on_next_segment] # 计算在下一线段上的x坐标 (Calculate x-coordinate on the next segment)
        closest_pt_y_next_np[mask_on_next_segment] = (next_start_np[:,1] + np.multiply(which_line_section_next_np,d_next))[mask_on_next_segment] # 计算在下一线段上的y坐标 (Calculate y-coordinate on the next segment)


        # 点到该最近点的最小距离（即点到线段的最近距离）(min dist from pt to that closest point (ie closes dist from pt to line segment))
        dist_next_np = np.sqrt((pt_np[:,self.x_index]-closest_pt_x_next_np)**2 + (pt_np[:,self.y_index]-closest_pt_y_next_np)**2) # 计算到下一线段的距离 (Calculate distance to next line segment)


        ############################################ 

        # 选择离哪个线段最近，并相应地更新变量 (pick which line segment it's closest to, and update vars accordingly)
        update_mask = dist_next_np <= min_perp_dist_np # 更新掩码 (Update mask)
        curr_seg_np[update_mask] += 1 # 更新当前线段 (Update current segment)
        moved_to_next_np[update_mask] = 1 # 更新移动到下一个标志 (Update moved to next flag)
        curr_forward_np[update_mask] = which_line_section_next_np[update_mask] # 更新当前前向 (Update current forward)
        min_perp_dist_np = np.minimum(min_perp_dist_np, dist_next_np) # 更新最小垂直距离 (Update minimum perpendicular distance)


        return min_perp_dist_np, curr_forward_np, curr_seg_np, moved_to_next_np # 返回几何量 (Return geometric quantities)