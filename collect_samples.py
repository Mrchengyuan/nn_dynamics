import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
# import rllab # rllab导入未使用，可以移除 (rllab import not used, can be removed)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)

class CollectSamples(object): # 定义CollectSamples类 (Define CollectSamples class)

    def __init__(self, env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories): # 定义构造函数 (Define constructor)
        self.env = env # 环境对象 (Environment object)
        self.policy = policy # 策略对象 (Policy object)
        self.visualize_at_all = visualize_rollouts # 是否可视化所有rollouts的标志 (Flag to visualize all rollouts)
        self.which_agent = which_agent # 智能体类型 (Agent type)

        self.low = self.env.action_space.low # 观测空间的下限 (Lower bound of observation space)
        self.high = self.env.action_space.high # 观测空间的上限 (Upper bound of observation space)
        self.shape = self.env.action_space.shape # 观测空间的形状 (Shape of observation space)

        # self.use_low = self.low + (self.high-self.low)/3.0 # 这些变量未使用，可以考虑移除 (These variables are unused, consider removing)
        # self.use_high = self.high - (self.high-self.low)/3.0 # 这些变量未使用，可以考虑移除 (These variables are unused, consider removing)

        self.dt_steps = dt_steps # 时间步长 (Time step duration)
        self.dt_from_xml = dt_from_xml # 从XML获取的dt值 (dt value from XML)

        self.follow_trajectories = follow_trajectories # 是否跟随轨迹的标志 (Flag to follow trajectories)
        
    def collect_samples(self, num_rollouts, steps_per_rollout): # 定义收集样本的函数 (Define function to collect samples)
        observations_list = [] # 初始化观测列表 (Initialize list for observations)
        actions_list = [] # 初始化动作列表 (Initialize list for actions)
        starting_states_list=[] # 初始化初始状态列表 (Initialize list for starting states) - 注意：Gymnasium的reset返回(obs,info)，"starting_state"的含义可能需要调整 (Note: Gymnasium's reset returns (obs,info), meaning of "starting_state" might need adjustment)
        rewards_list = [] # 初始化奖励列表 (Initialize list for rewards)
        visualization_frequency = 10 # 可视化频率 (Visualization frequency)

        for rollout_number in range(num_rollouts): # 对于每个rollout (For each rollout)
            # Gymnasium的reset不接受自定义参数如returnStartState, isSwimmer等。 (Gymnasium's reset doesn't take custom args like returnStartState, isSwimmer etc.)
            # 初始状态设置逻辑可能需要在环境自身或通过options参数处理 (Initial state setting logic might need to be handled in env itself or via options if supported)
            # starting_state 将是初始观测，或者如果环境允许，是更完整的状态表示 (starting_state will be the initial observation, or a more complete state representation if the env allows)

            reset_options = {} # 为reset传递的选项字典 (Options dictionary for reset)
            if self.which_agent == 2: # 如果是游泳者智能体 (If agent is Swimmer)
                 # 假设环境的reset方法可以处理这些选项 (Assuming env's reset can handle these options)
                reset_options['isSwimmer'] = True # 设置isSwimmer选项 (Set isSwimmer option)
                if self.follow_trajectories: # 如果跟随轨迹 (If following trajectories)
                    reset_options['need_diff_headings'] = True # 设置need_diff_headings选项 (Set need_diff_headings option)

            # Gymnasium env.reset() 返回 obs, info (Gymnasium env.reset() returns obs, info)
            observation, info = self.env.reset(options=reset_options if reset_options else None) # 重置环境并获取初始观测和信息 (Reset environment and get initial observation and info)
            # 假设starting_state是初始观测，如果需要完整的模拟器状态，则需要特定于环境的方法 (Assuming starting_state is the initial observation, if full simulator state is needed, env-specific methods would be required)
            starting_state = observation # 将初始观测作为starting_state (Use initial observation as starting_state)

            # 执行单个rollout (Perform a single rollout)
            observations, actions, reward_for_rollout = self.perform_rollout(observation, steps_per_rollout, 
                                                                        rollout_number, visualization_frequency) # 执行rollout (Perform rollout)

            rewards_list.append(reward_for_rollout) # 添加rollout奖励 (Add rollout reward)
            observations_arr = np.array(observations) # 将观测转换为numpy数组 (Convert observations to numpy array)
            actions_arr = np.array(actions) # 将动作转换为numpy数组 (Convert actions to numpy array)
            observations_list.append(observations_arr) # 添加观测到列表 (Add observations to list)
            actions_list.append(actions_arr) # 添加动作到列表 (Add actions to list)
            starting_states_list.append(starting_state) # 添加初始状态到列表 (Add starting state to list)

        # 返回列表，长度 = num_rollouts (return list of length = num rollouts)
        # 列表的每个条目包含一个rollout (each entry of that list contains one rollout)
        # 每个条目是 [steps_per_rollout x statespace_dim] 或 [steps_per_rollout x actionspace_dim] (each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim])
        return observations_list, actions_list, starting_states_list, rewards_list # 返回收集到的数据 (Return collected data)

    def perform_rollout(self, observation, steps_per_rollout, rollout_number, visualization_frequency): # 定义执行单个rollout的函数 (Define function to perform a single rollout)
        observations = [] # 初始化观测列表 (Initialize list for observations)
        actions = [] # 初始化动作列表 (Initialize list for actions)
        visualize = False # 是否可视化的标志 (Flag for visualization)
        reward_for_rollout = 0 # 初始化rollout奖励 (Initialize reward for rollout)

        if visualization_frequency > 0 and (rollout_number % visualization_frequency) == 0 : # 如果可视化频率大于0且是其倍数 (If visualization frequency is greater than 0 and it's a multiple)
            print("currently performing rollout #", rollout_number) # 打印当前rollout编号 (Print current rollout number)
            if(self.visualize_at_all): # 如果可视化所有rollouts (If visualizing all rollouts)
                all_states_vis=[] # 初始化所有状态列表（用于可视化）(Initialize list for all states (for visualization))
                print ("---- visualizing a rollout ----") # 打印可视化rollout信息 (Print message about visualizing rollout)
                visualize=True # 设置可视化标志为True (Set visualization flag to True)

        for step_num in range(steps_per_rollout): # 对于每个步数 (For each step)
            action, _ = self.policy.get_action(observation) # 从策略获取动作 (Get action from policy)

            observations.append(observation) # 添加当前观测 (Add current observation)
            actions.append(action) # 添加当前动作 (Add current action)

            # Gymnasium env.step() 返回 obs, reward, terminated, truncated, info (Gymnasium env.step() returns obs, reward, terminated, truncated, info)
            # collectingInitialData 参数不是标准Gymnasium API的一部分 (collectingInitialData parameter is not part of standard Gymnasium API)
            next_observation, reward, terminated, truncated, info = self.env.step(action) # 环境执行一步 (Environment takes a step)
            done = terminated or truncated # 组合终止和截断标志 (Combine terminated and truncated flags)
            reward_for_rollout+= reward # 累加奖励 (Accumulate reward)

            observation = np.copy(next_observation) # 更新观测 (Update observation)

            if done: # 如果完成 (If done)
                print("Had to stop rollout because a terminal or truncated state was reached.") # 打印终止信息 (Print termination message)
                break # 跳出循环 (Break the loop)

            if(visualize): # 如果可视化 (If visualizing)
                # 渲染应在环境创建时通过render_mode="human"启用 (Rendering should be enabled via render_mode="human" at env creation)
                # self.env.render() 调用可能不再需要，或者如果环境是以不同的render_mode创建的，则需要 (self.env.render() call might not be needed, or needed if env created with different render_mode)
                if self.env.render_mode == 'human': # 如果渲染模式是human (If render mode is human)
                    self.env.render() # 渲染环境 (Render environment)
                    time.sleep(self.dt_steps*self.dt_from_xml if self.dt_from_xml > 0 else 0.01) # 等待一段时间 (Wait for a period of time)
                elif self.which_agent == 0 and self.env.render_mode == 'rgb_array': # 如果是PointEnv且渲染模式是rgb_array (If agent is PointEnv and render mode is rgb_array)
                    # PointEnv的render在rgb_array模式下返回一个帧 (PointEnv's render returns a frame in rgb_array mode)
                    frame = self.env.render() # 获取渲染帧 (Get rendered frame)
                    if frame is not None: # 如果帧有效 (If frame is valid)
                         all_states_vis.append(frame) # 添加帧到列表 (Add frame to list)


        if(visualize and self.which_agent==0 and self.env.render_mode == 'rgb_array' and len(all_states_vis) > 0): # 如果可视化PointEnv的rgb_array (If visualizing PointEnv's rgb_array)
            # 此处的绘图逻辑可能需要调整，因为all_states_vis现在包含RGB图像帧 (Plotting logic here might need adjustment as all_states_vis now contains RGB image frames)
            # fig, ax = plt.subplots()
            # for frame_img in all_states_vis:
            # ax.imshow(frame_img) # 显示图像帧 (Display image frame)
            # plt.show() # 显示绘图 (Show plot)
            print("PointEnv RGB frames collected for visualization, plotting not implemented here.") # 打印信息 (Print message)
        elif visualize and self.which_agent==0 and self.env.render_mode == 'human': # 如果可视化PointEnv的human模式 (If visualizing PointEnv's human mode)
             # 人类模式渲染在step循环中处理 (Human mode rendering is handled in step loop)
            pass # 不执行任何操作 (Do nothing)
            
        return observations, actions, reward_for_rollout # 返回观测、动作和rollout奖励 (Return observations, actions, and rollout reward)