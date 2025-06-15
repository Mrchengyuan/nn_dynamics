import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import torch # 导入torch库 (Import torch library)
import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import os  # 导入os模块，用于检查环境变量 (Import os module for checking env vars)

# # 导入rllab环境 (import rllab envs) - rllab的normalize已被移除 (rllab's normalize has been removed)
# from rllab.envs.normalized_env import normalize
from point_env import PointEnv # 从point_env模块导入PointEnv类 (Import PointEnv class from point_env module) (已更新为Gymnasium) (Already updated to Gymnasium)


# 导入gymnasium环境 (import gymnasium envs)
import gymnasium as gym # 导入gymnasium库并重命名为gym (Import gymnasium and alias it as gym)


def add_noise(data_inp, noiseToSignal): # 定义添加噪声的函数 (Define function to add noise)
    data = copy.deepcopy(data_inp) # 深拷贝输入数据 (Deepcopy input data)
    mean_data = np.mean(data, axis=0) # 计算数据均值 (Calculate mean of data)
    std_of_noise = mean_data * noiseToSignal # 计算噪声标准差 (Calculate standard deviation of noise)
    for j in range(mean_data.shape[0]): # 遍历每个数据维度 (Iterate over each data dimension)
        if(std_of_noise[j] > 0): # 如果噪声标准差大于0 (If standard deviation of noise is greater than 0)
            data[:,j] = np.copy(data[:,j] + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],))) # 添加高斯噪声 (Add Gaussian noise)
    return data # 返回带噪声的数据 (Return noisy data)

def perform_rollouts(policy, num_rollouts, steps_per_rollout, visualize_rollouts, CollectSamples, 
                    env, which_agent, dt_steps, dt_from_xml, follow_trajectories): # 定义执行rollouts的函数 (Define function to perform rollouts)
    # 通过执行rollouts收集训练数据 (collect training data by performing rollouts)
    print("Beginning to do ", num_rollouts, " rollouts.") # 打印开始执行rollouts信息 (Print message about beginning rollouts)
    # CollectSamples类需要更新以使用Gymnasium API (CollectSamples class needs to be updated to use Gymnasium API)
    c = CollectSamples(env, policy, visualize_rollouts, which_agent, dt_steps, dt_from_xml, follow_trajectories) # 创建CollectSamples实例 (Create CollectSamples instance)
    states, controls, starting_states, rewards_list = c.collect_samples(num_rollouts, steps_per_rollout) # 收集样本 (Collect samples)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.") # 打印执行的rollout数量和步数 (Print number of rollouts and steps performed)
    return states, controls, starting_states, rewards_list # 返回状态、控制、初始状态和奖励列表 (Return states, controls, starting states, and rewards list)


def create_env(which_agent, seed=2, render_mode=None): # 定义创建环境的函数，添加seed和render_mode参数 (Define function to create environment, add seed and render_mode parameters)
    # 如果请求human渲染但系统没有DISPLAY，退回不渲染模式 (Fallback if DISPLAY missing)
    if render_mode == 'human' and os.environ.get('DISPLAY') is None:
        print("Warning: DISPLAY not found, falling back to headless mode")
        render_mode = None

    env = None # 初始化env为None (Initialize env to None)
    # 设置环境 (setup environment)
    if(which_agent==0): # 如果智能体类型为0 (If agent type is 0)
        env = PointEnv(render_mode=render_mode) # 创建PointEnv (Create PointEnv)
    elif(which_agent==1): # 如果智能体类型为1 (If agent type is 1)
        env = gym.make("Ant-v4", render_mode=render_mode) # 创建Ant环境 (Create Ant environment)
    elif(which_agent==2): # 如果智能体类型为2 (If agent type is 2)
        env = gym.make("Swimmer-v4", render_mode=render_mode) # 创建Swimmer环境 (Create Swimmer environment)
    elif(which_agent==3): # 如果智能体类型为3 (If agent type is 3)
        env = gym.make("Reacher-v4", render_mode=render_mode) # 创建Reacher环境 (Create Reacher environment)
    elif(which_agent==4): # 如果智能体类型为4 (If agent type is 4)
        env = gym.make("HalfCheetah-v4", render_mode=render_mode) # 创建HalfCheetah环境 (Create HalfCheetah environment)
    elif(which_agent==5): # 如果智能体类型为5 (If agent type is 5)
        raise NotImplementedError("RoachEnv is not defined or imported in helper_funcs.py") # 抛出未实现错误 (Raise NotImplementedError)
    elif(which_agent==6): # 如果智能体类型为6 (If agent type is 6)
        env = gym.make("Hopper-v4", render_mode=render_mode) # 创建Hopper环境 (Create Hopper environment)
    elif(which_agent==7): # 如果智能体类型为7 (If agent type is 7)
        env = gym.make("Walker2d-v4", render_mode=render_mode) # 创建Walker2d环境 (Create Walker2d environment)
    else: # 否则 (Otherwise)
        raise ValueError(f"Unknown agent type: {which_agent}") # 抛出未知智能体类型错误 (Raise error for unknown agent type)

    # rllab.envs.normalized_env.normalize 已移除。如果需要标准化，应使用Gymnasium的包装器。
    # (rllab.envs.normalized_env.normalize has been removed. If normalization is needed, Gymnasium wrappers should be used.)
    # 例如: env = gymnasium.wrappers.NormalizeObservation(env)
    # (e.g.: env = gymnasium.wrappers.NormalizeObservation(env))
    # print(f"Environment {type(env)} created. Normalization (if previously applied by rllab) is now removed.") # 打印环境创建信息 (Print environment creation message)


    # 从环境中获取dt值 (get dt value from env)
    dt_from_xml = 0.0 # 初始化dt_from_xml (Initialize dt_from_xml)
    try: # 尝试 (Try)
        # 尝试访问解包后的环境的dt属性 (Try to access dt attribute of the unwrapped environment)
        true_env = env.unwrapped if hasattr(env, 'unwrapped') else env # 获取真实环境 (Get the true environment)
        if hasattr(true_env, 'dt'): # 如果真实环境有dt属性 (If true environment has dt attribute)
             dt_from_xml = true_env.dt # 使用dt属性 (Use dt attribute)
        elif hasattr(true_env, 'model') and hasattr(true_env.model, 'opt') and hasattr(true_env.model.opt, 'timestep'): # 如果环境有model.opt.timestep属性 (If environment has model.opt.timestep attribute)
            dt_from_xml = true_env.model.opt.timestep # 获取MuJoCo环境的dt (Get dt for MuJoCo environment)
            # 在Gymnasium中，frame_skip通常包含在dt中，或者通过env.metadata['render_fps']间接获得 (In Gymnasium, frame_skip is often included in dt, or available via env.metadata['render_fps'])
        elif which_agent == 0: # PointEnv的特殊情况 (Special case for PointEnv)
            dt_from_xml = 0.1 # PointEnv使用固定的dt (PointEnv uses a fixed dt)
        else: # 否则 (Otherwise)
            print(f"Warning: Could not determine dt for {type(env)}. Using default 0.0") # 打印警告信息 (Print warning message)
    except Exception as e: # 捕获异常 (Catch exception)
        print(f"Warning: Error getting dt_from_xml for {type(env)}: {e}. Using default 0.0") # 打印错误信息 (Print error message)

    print("\n\n the dt is: ", dt_from_xml, "\n\n") # 打印dt值 (Print dt value)

    # 设置随机种子 (set seeds)
    np.random.seed(seed) # 设置NumPy的随机种子 (Set NumPy's random seed)
    torch.manual_seed(seed) # 设置PyTorch的CPU随机种子 (Set PyTorch's CPU random seed)
    if torch.cuda.is_available(): # 如果CUDA可用 (If CUDA is available)
        torch.cuda.manual_seed_all(seed) # 设置PyTorch所有GPU的随机种子 (Set PyTorch's random seed for all GPUs)
    # Gymnasium环境的种子在reset时或通过env.action_space.seed()设置 (Seed for gymnasium environments is set during reset or via env.action_space.seed())
    env.action_space.seed(seed) # 为动作空间设置种子 (Set seed for action space)

    gym.logger.min_level = gym.logger.WARN # 设置gymnasium日志级别 (Set gymnasium logger level)

    return env, dt_from_xml # 返回环境和dt值 (Return environment and dt value)


def visualize_rendering(starting_state, list_of_actions, env_inp, dt_steps, dt_from_xml, which_agent): # 定义可视化渲染函数 (Define function for visualizing rendering)
    env = env_inp #直接使用传入的环境 (Use the passed environment directly)

    # Gymnasium的reset返回 (obs, info) (Gymnasium's reset returns (obs, info))
    # 设置特定初始状态通常通过在reset的options中传递，或直接操纵模拟器状态（如果允许）
    # (Setting a specific initial state is usually done by passing it in options to reset, or manipulating simulator state directly if allowed)
    reset_options = {} # 初始化选项字典 (Initialize options dictionary)
    if starting_state is not None: # 如果提供了初始状态 (If starting_state is provided)
        # 实际的状态设置机制取决于环境实现 (Actual state setting mechanism depends on env implementation)
        # 对于Mujoco环境，可能需要设置qpos和qvel (For Mujoco envs, might need to set qpos and qvel)
        if hasattr(env.unwrapped, 'set_state') and callable(getattr(env.unwrapped, 'set_state')): # 如果有set_state方法 (If has set_state method)
            try: # 尝试 (Try)
                qpos_dim = env.unwrapped.model.nq # 获取qpos维度 (Get qpos dimension)
                qvel_dim = env.unwrapped.model.nv # 获取qvel维度 (Get qvel dimension)
                if len(starting_state) == qpos_dim + qvel_dim: # 如果长度匹配 (If length matches)
                    env.unwrapped.set_state(starting_state[:qpos_dim], starting_state[qpos_dim:]) # 设置状态 (Set state)
                obs, info = env.reset(options=reset_options) # 重置以应用状态 (Reset to apply state)
            except Exception as e: # 捕获异常 (Catch exception)
                print(f"Warning: Failed to set state for visualization using env.unwrapped.set_state: {e}. Using default reset.") # 打印警告 (Print warning)
                obs, info = env.reset(options=reset_options) # 使用默认重置 (Use default reset)
        else: # 否则 (Otherwise)
             print(f"Warning: env.unwrapped.set_state not available. Visualization might start from a default state.") # 打印警告 (Print warning)
             obs, info = env.reset(options=reset_options) # 使用默认重置 (Use default reset)
    else: # 否则 (Otherwise)
        obs, info = env.reset(options=reset_options) # 使用默认重置 (Use default reset)


    for action in list_of_actions: # 对于动作列表中的每个动作 (For each action in the list of actions)
        action_to_take = action # 要执行的动作 (Action to take)
        if isinstance(action, np.ndarray) and action.ndim > 1 and action.shape[0] == 1: # 如果动作是形状为(1,N)的numpy数组 (If action is a numpy array of shape (1,N))
            action_to_take = action[0] # 取第一个元素 (Take the first element)

        # Gymnasium的step返回 (obs, reward, terminated, truncated, info) (Gymnasium's step returns (obs, reward, terminated, truncated, info))
        next_obs, reward, terminated, truncated, info = env.step(action_to_take) # collectingInitialData参数已移除 (collectingInitialData parameter removed)
        done = terminated or truncated # 完成标志 (Done flag)

        if hasattr(env, 'render_mode') and env.render_mode == 'human': # 如果渲染模式是human (If render mode is human)
             env.render() # 渲染环境 (Render the environment)
             time.sleep(dt_steps * dt_from_xml if dt_from_xml > 0 else 0.01) # 等待一段时间 (Wait for a period of time)

        if done: # 如果完成 (If done)
            break # 跳出循环 (Break loop)

    print("Done rendering.") # 打印渲染完成信息 (Print message that rendering is done)
    return # 返回 (Return)