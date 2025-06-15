import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)

class Policy_Random(object): # 定义Policy_Random类 (Define Policy_Random class)

    def __init__(self, env): # 定义构造函数 (Define constructor)

        # 变量 (vars)
        self.env = env # 环境对象 (Environment object)
        self.low_val = self.env.action_space.low # 动作空间的下限 (Lower bound of action space)
        self.high_val = self.env.action_space.high # 动作空间的上限 (Upper bound of action space)
        self.shape = self.env.action_space.shape # 动作空间的形状 (Shape of action space)
        print("Created a random policy, where actions are selected between ", self.low_val, ", and ", self.high_val) # 打印创建随机策略的信息 (Print message about creating a random policy)
        
    def get_action(self, observation): # 定义获取动作的函数 (Define function to get action)
        # 在动作空间的下限和上限之间均匀采样动作，agent_infos通常为0或空字典 (Sample action uniformly between the lower and upper bounds of the action space, agent_infos is usually 0 or an empty dict)
        return np.random.uniform(self.low_val, self.high_val, self.shape), {} # 返回随机动作和空的agent_info (Return random action and empty agent_info)