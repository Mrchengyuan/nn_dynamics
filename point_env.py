import gymnasium as gym # 导入gymnasium库并重命名为gym (Import gymnasium and alias it as gym)
from gymnasium import spaces # 从gymnasium导入spaces模块 (Import spaces module from gymnasium)
import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)

class PointEnv(gym.Env): # 定义PointEnv类，继承自gym.Env (Define PointEnv class, inheriting from gym.Env)
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # 定义元数据，包括渲染模式和FPS (Define metadata, including render modes and FPS)

    def __init__(self, render_mode=None): # 定义构造函数，添加render_mode参数 (Define constructor, add render_mode parameter)
        super(PointEnv, self).__init__() # 调用父类的构造函数 (Call parent class constructor)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64) # 状态空间 = [x, y, vx, vy] (state space = [x, y, vx, vy])
        self.action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float64) # 控制是施加到质点上的力 (controls are the forces applied to pointmass)

        self._state = None # 初始化内部状态为None (Initialize internal state to None)
        self.render_mode = render_mode # 存储渲染模式 (Store render mode)

        # 如果使用可视化渲染，可能需要初始化可视化组件 (If visual rendering is used, visualization components might need initialization)
        # 例如: self.window = None; self.clock = None (e.g. self.window = None; self.clock = None)

    def reset(self, seed=None, options=None, init_state=None): # 定义重置环境的函数，更新签名以匹配gymnasium (Define function to reset the environment, update signature to match gymnasium)
        super().reset(seed=seed) # 调用父类的reset方法处理种子 (Call parent class's reset method to handle seed)

        if init_state is None: # 如果未提供初始状态 (If initial state is not provided)
            self._state = np.zeros((4,), dtype=np.float64) # 初始化状态为零数组 (Initialize state as a zero array)
            self._state[0] = self.np_random.uniform(-10, 10) # 使用gymnasium的np_random初始化x坐标 (Initialize x coordinate using gymnasium's np_random)
            self._state[1] = self.np_random.uniform(-10, 10) # 使用gymnasium的np_random初始化y坐标 (Initialize y coordinate using gymnasium's np_random)
            self._state[2:] = 0.0 # 初始化速度为0 (Initialize velocities to 0)
        else: # 否则 (Otherwise)
            self._state = np.array(init_state, dtype=np.float64) # 使用提供的初始状态 (Use the provided initial state)

        observation = np.copy(self._state) # 复制当前状态作为观测 (Copy current state as observation)
        info = {} # 初始化信息字典 (Initialize info dictionary)

        if self.render_mode == "human": # 如果渲染模式是human (If render mode is human)
            self._render_frame() # 渲染当前帧 (Render current frame)

        return observation, info # 返回观测和信息字典 (Return observation and info dictionary)

    def step(self, action): # 定义环境执行一步的函数 (Define function for the environment to take a step)
        action = np.asarray(action, dtype=np.float64) # 确保动作为numpy数组和正确类型 (Ensure action is numpy array and correct type)
        # 下一个状态 = 执行“动作”后发生的情况 (next state = what happens after taking "action")
        temp_state = np.copy(self._state) # 复制当前状态到临时状态 (Copy current state to temporary state)
        dt = 0.1 # 时间步长 (Time step)

        # 物理更新 (Physics update)
        temp_state[0] = self._state[0] + self._state[2]*dt + 0.5*action[0]*dt*dt # 更新x坐标 (Update x coordinate)
        temp_state[1] = self._state[1] + self._state[3]*dt + 0.5*action[1]*dt*dt # 更新y坐标 (Update y coordinate)
        temp_state[2] = self._state[2] + action[0]*dt # 更新x方向速度 (Update velocity in x direction)
        temp_state[3] = self._state[3] + action[1]*dt # 更新y方向速度 (Update velocity in y direction)
        self._state = np.copy(temp_state) # 更新当前状态 (Update current state)

        # 设置你关心的奖励 (make the reward what you care about)
        x, y, vx, vy = self._state # 解包状态 (Unpack state)
        reward = vx - np.sqrt(abs(y-0)) # 我们关心的是在x方向前进...并保持y值接近0...（即“直行”）(we care about moving in the forward x direction... and keeping our y value close to 0... (aka "going straight"))

        # Gymnasium使用terminated和truncated (Gymnasium uses terminated and truncated)
        terminated = False # 终止标志 (Terminated flag)
        # done = 0 #x>500 # 何时认为完成（rollout停止...“终止”）(when do you consider this to be "done" (rollout stops... "terminal"))
        # if x > 500: terminated = True # 例如，如果x超过500则终止 (Example: if x > 500, then terminate)

        truncated = False # 截断标志 (Truncated flag) - 例如，如果达到了最大步数 (e.g., if max steps are reached)

        next_observation = np.copy(self._state) # 复制下一个状态作为观测 (Copy next state as observation)
        info = {} # 初始化信息字典 (Initialize info dictionary)

        if self.render_mode == "human": # 如果渲染模式是human (If render mode is human)
            self._render_frame() # 渲染当前帧 (Render current frame)

        return next_observation, reward, terminated, truncated, info # 返回观测、奖励、终止标志、截断标志和信息字典 (Return observation, reward, terminated flag, truncated flag, and info dictionary)

    def render(self): # 定义渲染函数 (Define render function)
        if self.render_mode == 'rgb_array': # 如果渲染模式是rgb_array (If render mode is rgb_array)
            return self._render_frame() # 返回渲染帧 (Return rendered frame)
        elif self.render_mode == 'human': # 如果渲染模式是human (If render mode is human)
            # Human rendering is handled in step/reset or by a separate window managed by _render_frame
            # 人类渲染在step/reset中处理，或由_render_frame管理的单独窗口处理
            pass # 不执行任何操作 (Do nothing)
        # return self._state # 旧的返回状态逻辑 (Old logic of returning state)

    def _render_frame(self): # 定义内部渲染帧函数 (Define internal render frame function)
        if self.render_mode == 'human': # 如果渲染模式是human (If render mode is human)
            # 在此处实现人类可读的渲染，例如使用pygame (Implement human-readable rendering here, e.g., using pygame)
            # print(f"Current state: x={self._state[0]:.2f}, y={self._state[1]:.2f}, vx={self._state[2]:.2f}, vy={self._state[3]:.2f}") # 打印当前状态 (Print current state)
            pass # 暂时不实现可视化 (Temporarily do not implement visualization)
        elif self.render_mode == 'rgb_array': # 如果渲染模式是rgb_array (If render mode is rgb_array)
            # 返回代表当前状态的RGB图像 (Return an RGB image representing the current state)
            # 这需要一个实际的渲染实现 (This requires an actual rendering implementation)
            # 例如，使用matplotlib创建一个图像 (For example, create an image using matplotlib)
            # fig, ax = plt.subplots()
            # ax.scatter(self._state[0], self._state[1], s=100, c='blue')
            # ax.set_xlim([-15, 15])
            # ax.set_ylim([-15, 15])
            # fig.canvas.draw()
            # img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close(fig)
            # return img_data
            return np.zeros((100, 100, 3), dtype=np.uint8) # 返回一个占位符黑色图像 (Return a placeholder black image)
        return None # 默认不返回任何内容 (Default return None)

    def close(self): # 定义关闭环境函数 (Define close environment function)
        # 在此处清理任何打开的资源，例如渲染窗口 (Clean up any open resources here, e.g., rendering windows)
        pass # 暂时无操作 (No operation for now)