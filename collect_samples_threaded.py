import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
# import copy # 重复导入copy模块，可以移除 (Repeated import of copy module, can be removed)
import multiprocessing # 导入multiprocessing模块，用于多进程处理 (Import multiprocessing module for multi-processing)

class CollectSamples(object): # 定义CollectSamples类 (Define CollectSamples class)

    def __init__(self, env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories): # 定义构造函数 (Define constructor)
        self.main_env = copy.deepcopy(env) # 深拷贝主环境对象 (Deepcopy the main environment object)
        self.policy = policy # 策略对象 (Policy object)
        self.visualize_at_all = visualize_rollouts # 是否可视化所有rollouts的标志 (Flag to visualize all rollouts)
        self.which_agent = which_agent # 智能体类型 (Agent type)
        self.list_observations=[] # 初始化观测列表 (Initialize list for observations)
        self.list_actions=[] # 初始化动作列表 (Initialize list for actions)
        self.list_starting_states=[] # 初始化初始状态列表 (Initialize list for starting states)

        self.stateDim = self.main_env.observation_space.shape[0] # 状态空间维度 (Dimension of state space)
        self.actionDim = self.main_env.action_space.shape[0] # 动作空间维度 (Dimension of action space)

        self.dt_steps = dt_steps # 时间步长 (Time step duration)
        self.dt_from_xml = dt_from_xml # 从XML获取的dt值 (dt value from XML)
        self.follow_trajectories = follow_trajectories # 是否跟随轨迹的标志 (Flag to follow trajectories)

    def collect_samples(self, num_rollouts, steps_per_rollout): # 定义收集样本的函数 (Define function to collect samples)
        
        # 变量 (vars)
        # all_processes=[] # all_processes未使用，可以移除 (all_processes is unused, can be removed)
        # 可视化频率，避免num_rollouts为0时出错 (Visualization frequency, avoid error if num_rollouts is 0)
        visualization_frequency = num_rollouts/10 if num_rollouts > 0 else 0 # 可视化频率 (Visualization frequency)
        # num_workers=multiprocessing.cpu_count() # 检测核心数量 (detect number of cores) - 未使用 (unused)
        # 根据需要设置进程池大小，例如使用cpu_count()或固定值 (Set pool size as needed, e.g., using cpu_count() or a fixed value)
        pool_size = min(8, multiprocessing.cpu_count()) # 设置进程池大小，最多8个或CPU核心数 (Set pool size, max 8 or number of CPU cores)
        if pool_size <= 0: # 如果进程池大小小于等于0 (If pool size is less than or equal to 0)
            pool_size = 1 # 设置为1 (Set to 1)

        # 在Windows上，多进程池的创建可能需要在 if __name__ == '__main__': 块中 (On Windows, multiprocessing Pool creation might need to be in if __name__ == '__main__': block)
        # 为简单起见，此处保持不变，但如果目标平台是Windows，则可能需要调整 (For simplicity, keeping it here, but may need adjustment if targeting Windows)
        try: # 尝试 (Try)
            pool = multiprocessing.Pool(pool_size) # 创建进程池 (Create process pool)

            # 用于运行rollouts的多进程处理（利用多个核心）(multiprocessing for running rollouts (utilize multiple cores))
            async_results = [] # 存储异步结果 (Store asynchronous results)
            for rollout_number in range(num_rollouts): # 对于每个rollout (For each rollout)
                result = pool.apply_async(self.do_rollout, # 异步应用do_rollout函数 (Asynchronously apply do_rollout function)
                                        args=(steps_per_rollout, rollout_number, visualization_frequency), # 传递参数 (Pass arguments)
                                        callback=self.mycallback) # 设置回调函数 (Set callback function)
                async_results.append(result) # 添加异步结果到列表 (Add async result to list)

            for result in async_results: # 对于每个异步结果 (For each async result)
                result.wait() # 等待结果完成 (Wait for result to complete)


        finally: # 最终 (Finally)
            pool.close() # 不再向进程池添加任务 (Not going to add anything else to the pool)
            pool.join() # 等待所有进程终止 (Wait for the processes to terminate)


        # 返回列表，长度 = num_rollouts (return lists of length = num rollouts)
        # 每个条目包含一个rollout (each entry contains one rollout)
        # 每个条目是 [steps_per_rollout x statespace_dim] 或 [steps_per_rollout x actionspace_dim] (each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim])
        return self.list_observations, self.list_actions, self.list_starting_states, [] # 返回收集到的数据和空奖励列表 (Return collected data and an empty rewards list)

    def mycallback(self, x): # 定义回调函数 (Define callback function) x的形状是 [numSteps, state + action + starting_state_repeated] (x's shape is [numSteps, state + action + starting_state_repeated])
        if x.shape[0] > 0: # 确保x不为空 (Ensure x is not empty)
            self.list_observations.append(x[:,0:self.stateDim]) # 添加观测数据 (Add observation data)
            self.list_actions.append(x[:,self.stateDim:(self.stateDim+self.actionDim)]) # 添加动作数据 (Add action data)
            # 假设starting_state被重复附加到每一行 (Assuming starting_state was appended repeatedly to each row)
            # 并且所有行的starting_state都相同，所以只取第一行的 (And all rows have the same starting_state, so take from the first row)
            self.list_starting_states.append(x[0,(self.stateDim+self.actionDim):]) # 添加初始状态数据 (Add starting state data)
        # 如果x为空（例如，num_steps_taken为0），则不添加任何内容 (If x is empty (e.g. num_steps_taken was 0), add nothing)


    def do_rollout(self, steps_per_rollout, rollout_number, visualization_frequency): # 定义执行单个rollout的函数 (Define function to perform a single rollout)
        # 初始化变量 (init vars)
        #print("START ", rollout_number) # 打印开始rollout编号 (Print start rollout number)
        observations = [] # 初始化观测列表 (Initialize list for observations)
        actions = [] # 初始化动作列表 (Initialize list for actions)
        visualize = False # 是否可视化的标志 (Flag for visualization)

        # 每个进程创建自己的环境副本 (Each process creates its own copy of the environment)
        # 注意：如果环境本身不可picklable（例如，包含某些类型的锁或外部资源），则此方法可能会失败 (Note: this might fail if the environment itself is not picklable e.g. contains certain types of locks or external resources)
        env = copy.deepcopy(self.main_env) # 深拷贝主环境对象 (Deepcopy the main environment object)

        # 重置环境 (reset env) - Gymnasium API
        reset_options = {} # 为reset传递的选项字典 (Options dictionary for reset)
        if self.which_agent == 2: # 如果是游泳者智能体 (If agent is Swimmer)
            reset_options['isSwimmer'] = True # 设置isSwimmer选项 (Set isSwimmer option)
            if self.follow_trajectories: # 如果跟随轨迹 (If following trajectories)
                reset_options['need_diff_headings'] = True # 设置need_diff_headings选项 (Set need_diff_headings option)

        # Gymnasium env.reset() 返回 obs, info (Gymnasium env.reset() returns obs, info)
        observation, info = env.reset(options=reset_options if reset_options else None) # 重置环境 (Reset environment)
        starting_state = observation # 将初始观测作为starting_state (Use initial observation as starting_state)


        # 仅有时可视化 (visualize only sometimes)
        if(visualization_frequency > 0 and (rollout_number%visualization_frequency)==0): # 如果可视化频率大于0且是其倍数 (If visualization frequency is greater than 0 and it's a multiple)
            if(self.visualize_at_all): # 如果可视化所有rollouts (If visualizing all rollouts)
                all_states_vis=[] # 初始化所有状态列表（用于可视化）(Initialize list for all states (for visualization))
                print ("---- visualizing a rollout ----") # 打印可视化rollout信息 (Print message about visualizing rollout)
                visualize=True # 设置可视化标志为True (Set visualization flag to True)

        for step_num in range(steps_per_rollout): # 对于每个步数 (For each step)

            # 决定采取什么动作 (decide what action to take)
            action, _ = self.policy.get_action(observation) # 从策略获取动作 (Get action from policy)

            # 跟踪观测和动作 (keep tracks of observations + actions)
            observations.append(observation) # 添加当前观测 (Add current observation)
            actions.append(action) # 添加当前动作 (Add current action)

            # 执行动作 (perform the action) - Gymnasium API
            # collectingInitialData 参数不是标准Gymnasium API的一部分 (collectingInitialData parameter is not part of standard Gymnasium API)
            next_observation, reward, terminated, truncated, info = env.step(action) # 环境执行一步 (Environment takes a step)
            done = terminated or truncated # 组合终止和截断标志 (Combine terminated and truncated flags)


            # 更新观测 (update the observation)
            observation = np.copy(next_observation) # 复制下一个观测 (Copy next observation)
            
            if done: # 如果完成 (If done)
                #print("Had to stop rollout because terminal state was reached.") # 打印终止信息 (Print termination message)
                break # 跳出循环 (Break the loop)

            if(visualize): # 如果可视化 (If visualizing)
                if self.main_env.render_mode == 'human': # 如果渲染模式是human (If render mode is human)
                    env.render() # 渲染环境 (Render the environment)
                    time.sleep(self.dt_steps*self.dt_from_xml if self.dt_from_xml > 0 else 0.01) # 等待一段时间 (Wait for a period of time)
                elif self.which_agent == 0 and self.main_env.render_mode == 'rgb_array': # 如果是PointEnv且渲染模式是rgb_array (If agent is PointEnv and render mode is rgb_array)
                    frame = env.render() # 获取渲染帧 (Get rendered frame)
                    if frame is not None: # 如果帧有效 (If frame is valid)
                        all_states_vis.append(frame) # 添加帧到列表 (Add frame to list)


        if(visualize and self.which_agent==0 and self.main_env.render_mode == 'rgb_array' and len(all_states_vis) > 0): # 如果可视化PointEnv的rgb_array (If visualizing PointEnv's rgb_array)
            # 此处的绘图逻辑可能需要调整 (Plotting logic here might need adjustment)
            print("PointEnv RGB frames collected for visualization in thread, plotting in main thread if needed.") # 打印信息 (Print message)


        if(visualization_frequency > 0 and (rollout_number%visualization_frequency)==0): # 如果可视化频率大于0且是其倍数 (If visualization frequency is greater than 0 and it's a multiple)
            print("Completed rollout # ", rollout_number) # 打印完成rollout编号 (Print completed rollout number)

        # 将初始状态平铺以匹配动作/观测的数量，以便在回调中轻松访问 (Tile starting_state to match number of actions/observations for easy access in callback)
        num_steps_taken = len(actions) # 获取执行的步数 (Get number of steps taken)
        if num_steps_taken == 0: # 如果没有执行任何步骤 (If no steps were taken)
            return np.zeros((0, self.stateDim + self.actionDim + self.stateDim)) # 返回一个空的二维数组 (Return an empty 2D array)

        array_starting_state = np.tile(starting_state, (num_steps_taken,1)) # 平铺初始状态 (Tile starting state)
        return np.concatenate((np.array(observations), np.array(actions), array_starting_state), axis=1) # 连接观测、动作和初始状态 (Concatenate observations, actions, and starting state)