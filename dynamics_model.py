
import numpy as np # 导入numpy库，用于数值计算
import numpy.random as npr # 导入numpy的随机数生成模块
import torch # 导入torch库
import torch.nn as nn # 导入torch.nn模块，用于构建神经网络
import time # 导入time模块，用于时间相关操作
import math # 导入math模块，用于数学运算

from feedforward_network import FeedforwardNetwork # 从feedforward_network模块导入FeedforwardNetwork类


class Dyn_Model(nn.Module): # 定义一个名为Dyn_Model的类，继承自nn.Module

    def __init__(self, inputSize, outputSize, learning_rate, batchsize, which_agent, x_index, y_index,
                num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, print_minimal, device='cpu'): # 定义构造函数
        super(Dyn_Model, self).__init__() # 调用父类的构造函数

        # 初始化变量 (init vars)
        self.batchsize = batchsize # 批处理大小
        self.which_agent = which_agent # 指示哪个智能体
        self.x_index = x_index # x索引
        self.y_index = y_index # y索引
        self.inputSize = inputSize # 输入大小
        self.outputSize = outputSize # 输出大小
        # 将均值和标准差转换为PyTorch张量并移到指定设备 (convert means and stds to PyTorch tensors and move to device)
        self.mean_x = torch.tensor(mean_x, dtype=torch.float32, device=device) # x的均值
        self.mean_y = torch.tensor(mean_y, dtype=torch.float32, device=device) # y的均值
        self.mean_z = torch.tensor(mean_z, dtype=torch.float32, device=device) # z的均值
        self.std_x = torch.tensor(std_x, dtype=torch.float32, device=device) # x的标准差
        self.std_y = torch.tensor(std_y, dtype=torch.float32, device=device) # y的标准差
        self.std_z = torch.tensor(std_z, dtype=torch.float32, device=device) # z的标准差
        self.print_minimal = print_minimal # 是否最小化打印信息
        self.device = device # 指定设备 (cpu 或 cuda)

        # 前向传播网络 (forward pass network)
        self.ff_network = FeedforwardNetwork(self.inputSize, self.outputSize,
                                             num_fc_layers, depth_fc_layers).to(device) # 创建前馈网络实例并移到指定设备

        # 损失函数 (loss function)
        self.loss_fn = nn.MSELoss() # 使用均方误差损失

        # 优化器 (optimizer)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) # 使用Adam优化器

    def forward(self, x): # 定义前向传播函数
        return self.ff_network(x) # 通过前馈网络传递输入x

    # 注意：原始的train方法包含TensorFlow特有的训练循环和会话管理。
    # 在PyTorch中，训练循环通常在外部管理。
    # 此处保留方法名，但其功能将调整为执行一步训练（计算损失、反向传播、更新参数）。
    # 完整的epoch迭代和数据加载逻辑应在外部处理。
    # (Note: The original train method contained TensorFlow-specific training loops and session management.)
    # (In PyTorch, the training loop is typically managed externally.)
    # (The method name is kept here, but its functionality will be adjusted to perform one step of training (calculate loss, backpropagate, update parameters).)
    # (Complete epoch iteration and data loading logic should be handled externally.)
    def train_step(self, dataX_batch, dataZ_batch): # 定义训练步骤函数
        # 将Numpy数据转换为PyTorch张量并移到指定设备 (Convert Numpy data to PyTorch tensors and move to device)
        dataX_batch_torch = torch.tensor(dataX_batch, dtype=torch.float32, device=self.device) # 输入数据批次
        dataZ_batch_torch = torch.tensor(dataZ_batch, dtype=torch.float32, device=self.device) # 标签数据批次

        self.optimizer.zero_grad() # 清除之前的梯度 (Clear previous gradients)
        predictions = self.forward(dataX_batch_torch) # 进行前向传播，获取预测结果 (Perform forward pass to get predictions)
        loss = self.loss_fn(predictions, dataZ_batch_torch) # 计算损失 (Calculate loss)
        loss.backward() # 反向传播，计算梯度 (Backpropagate to calculate gradients)
        self.optimizer.step() # 更新网络参数 (Update network parameters)
        return loss.item() # 返回损失值 (Return the loss value)

    # 验证方法，在PyTorch中，这通常只计算损失，不执行反向传播或优化 (Validation method, in PyTorch this usually just calculates loss, no backpropagation or optimization)
    def run_validation(self, inputs, outputs): # 定义验证运行函数
        # 将Numpy数据转换为PyTorch张量并移到指定设备 (Convert Numpy data to PyTorch tensors and move to device)
        inputs_torch = torch.tensor(inputs, dtype=torch.float32, device=self.device) # 输入数据
        outputs_torch = torch.tensor(outputs, dtype=torch.float32, device=self.device) # 输出数据

        with torch.no_grad(): # 在此块中不计算梯度 (Do not calculate gradients in this block)
            predictions = self.forward(inputs_torch) # 进行前向传播，获取预测结果 (Perform forward pass to get predictions)
            loss = self.loss_fn(predictions, outputs_torch) # 计算损失 (Calculate loss)

        if not self.print_minimal: # 如果不是最小化打印
            print ("Validation set size: ", inputs.shape[0]) # 打印验证集大小
            print ("Validation set's total loss: ", loss.item()) # 打印验证集总损失

        return loss.item() # 返回损失值

    # 使用学习到的动力学模型在每个步骤进行多步预测 (Multistep prediction using the learned dynamics model at each step)
    def do_forward_sim(self, forwardsim_x_true, forwardsim_y, many_in_parallel, env_inp=None, which_agent=None): # 定义前向模拟函数
        # (env_inp and which_agent are not used in the PyTorch version based on original TF logic, but kept for signature consistency if needed later)
        # (env_inp 和 which_agent 在基于原始TF逻辑的PyTorch版本中未使用，但为保持签名一致性而保留，以备后用)
        state_list = [] # 初始化状态列表 (Initialize state list)

        # 将初始状态和控制输入转换为PyTorch张量 (Convert initial states and control inputs to PyTorch tensors)
        forwardsim_x_true_torch = torch.tensor(forwardsim_x_true, dtype=torch.float32, device=self.device) # 真实x的模拟输入
        forwardsim_y_torch = torch.tensor(forwardsim_y, dtype=torch.float32, device=self.device) # y的模拟输入

        if many_in_parallel: # 如果并行处理多个模拟 (If processing multiple simulations in parallel)
            N = forwardsim_y_torch.shape[0] # 模拟数量 (Number of simulations)
            horizon = forwardsim_y_torch.shape[1] # 模拟步长 (Simulation horizon)

            # 扩展均值和标准差张量以匹配批次大小 (Expand mean and std tensors to match batch size)
            array_stdz = self.std_z.unsqueeze(0).repeat(N, 1) # z的标准差数组
            array_meanz = self.mean_z.unsqueeze(0).repeat(N, 1) # z的均值数组
            array_stdy = self.std_y.unsqueeze(0).repeat(N, 1) # y的标准差数组
            array_meany = self.mean_y.unsqueeze(0).repeat(N, 1) # y的均值数组
            array_stdx = self.std_x.unsqueeze(0).repeat(N, 1) # x的标准差数组
            array_meanx = self.mean_x.unsqueeze(0).repeat(N, 1) # x的均值数组

            if forwardsim_x_true_torch.ndim == 1: # 如果forwardsim_x_true_torch是一维的 (If forwardsim_x_true_torch is 1D)
                 # N个起始状态，每个并行模拟一个 (N starting states, one for each simultaneous sim)
                curr_states = forwardsim_x_true_torch.unsqueeze(0).repeat(N, 1) # 当前状态
            else: # 否则 (Otherwise)
                curr_states = forwardsim_x_true_torch.clone() # 复制当前状态 (Clone current states)

            # 逐个时间步推进所有N个模拟 (Advance all N sims, one timestep at a time)
            for timestep in range(horizon): # 对于每个时间步
                state_list.append(curr_states.cpu().numpy().copy()) # 将当前状态（转换为numpy）添加到列表 (Add current states (converted to numpy) to the list)

                # 预处理状态和动作 (Preprocess states and actions)
                states_preprocessed = (curr_states - array_meanx) / (array_stdx + 1e-8) # 预处理后的状态 (Preprocessed states, add epsilon to std to avoid division by zero)
                actions_preprocessed = (forwardsim_y_torch[:, timestep, :] - array_meany) / (array_stdy + 1e-8) # 预处理后的动作 (Preprocessed actions, add epsilon)
                inputs_list = torch.cat((states_preprocessed, actions_preprocessed), dim=1) # 拼接状态和动作作为输入 (Concatenate states and actions as input)

                with torch.no_grad(): # 不计算梯度 (Do not calculate gradients)
                    model_output = self.forward(inputs_list) # 模型输出 (Model output)

                state_differences = model_output * array_stdz + array_meanz # 计算状态差异 (Calculate state differences)
                curr_states = curr_states + state_differences # 更新当前状态 (Update current states)
            
            state_list.append(curr_states.cpu().numpy().copy()) # 添加最终状态 (Add final states)
        else: # 如果不并行处理 (If not processing in parallel)
            curr_state = forwardsim_x_true_torch[0].clone() # 当前状态 (Current state)

            for i in range(forwardsim_y_torch.shape[0]): # 对于每个控制输入
                curr_control = forwardsim_y_torch[i] # 当前控制 (Current control)
                state_list.append(curr_state.cpu().numpy().copy()) # 添加当前状态到列表 (Add current state to list)
                curr_control_expanded = curr_control.unsqueeze(0) # 扩展控制维度 (Expand control dimension)

                # 预处理状态和控制 (Preprocess state and control)
                curr_state_preprocessed = (curr_state - self.mean_x) / (self.std_x + 1e-8) # 预处理后的当前状态 (Preprocessed current state)
                curr_control_preprocessed = (curr_control_expanded - self.mean_y) / (self.std_y + 1e-8) # 预处理后的当前控制 (Preprocessed current control)

                #确保curr_state_preprocessed是1D的，如果它是0D或者更高维度 (Ensure curr_state_preprocessed is 1D if it's 0D or higher)
                if curr_state_preprocessed.ndim == 0: # 如果当前状态预处理后是0维
                    curr_state_preprocessed = curr_state_preprocessed.unsqueeze(0) # 增加一个维度
                elif curr_state_preprocessed.ndim > 1: # 如果当前状态预处理后维度大于1
                     curr_state_preprocessed = curr_state_preprocessed.squeeze() # 压缩维度

                if curr_control_preprocessed.ndim > 1: # 如果当前控制预处理后维度大于1
                    curr_control_preprocessed = curr_control_preprocessed.squeeze(0) # 压缩第一个维度

                inputs_preprocessed = torch.cat((curr_state_preprocessed, curr_control_preprocessed), dim=0).unsqueeze(0) # 拼接并扩展输入维度 (Concatenate and expand input dimension)


                with torch.no_grad(): # 不计算梯度 (Do not calculate gradients)
                    model_output = self.forward(inputs_preprocessed) # 模型输出 (Model output)

                state_differences = model_output.squeeze(0) * self.std_z + self.mean_z # 计算状态差异 (Calculate state differences)
                next_state = curr_state + state_differences # 计算下一状态 (Calculate next state)
                curr_state = next_state.clone() # 更新当前状态 (Update current state)

            state_list.append(curr_state.cpu().numpy().copy()) # 添加最终状态 (Add final state)

        return state_list # 返回状态列表 (Return state list)

    # 加载模型权重的方法 (Method to load model weights)
    def load_weights(self, path): # 定义加载权重函数
        self.load_state_dict(torch.load(path, map_location=self.device)) # 从指定路径加载模型状态字典
        print(f"Model weights loaded from {path}") # 打印模型权重已加载信息

    # 保存模型权重的方法 (Method to save model weights)
    def save_weights(self, path): # 定义保存权重函数
        torch.save(self.state_dict(), path) # 将模型状态字典保存到指定路径
        print(f"Model weights saved to {path}") # 打印模型权重已保存信息
