import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)

class RewardFunctions: # 定义RewardFunctions类 (Define RewardFunctions class)

    def __init__(self, which_agent, x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, 
                frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index): # 定义构造函数 (Define constructor)
        self.which_agent = which_agent # 智能体类型 (Agent type)
        self.x_index = x_index # x坐标索引 (x-coordinate index)
        self.y_index = y_index # y坐标索引 (y-coordinate index)
        self.z_index = z_index  # z坐标索引 (z-coordinate index)
        self.yaw_index = yaw_index  # 偏航角索引 (Yaw angle index)
        self.joint1_index = joint1_index  # 关节1索引 (Joint 1 index)
        self.joint2_index = joint2_index  # 关节 2索引 (Joint 2 index)
        self.frontleg_index = frontleg_index # 前腿索引 (Front leg index)
        self.frontshin_index = frontshin_index # 前胫索引 (Front shin index)
        self.frontfoot_index = frontfoot_index # 前脚索引 (Front foot index)
        self.xvel_index = xvel_index # x方向速度索引 (x-velocity index)
        self.orientation_index = orientation_index # 方向索引 (Orientation index)

    def get_reward_func(self, follow_trajectories, desired_states, horiz_penalty_factor, 
                        forward_encouragement_factor, heading_penalty_factor): # 定义获取奖励函数的函数 (Define function to get reward function)

        # 初始化变量 (init vars)
        self.desired_states= desired_states # 期望状态 (Desired states)
        self.horiz_penalty_factor = horiz_penalty_factor # 水平惩罚因子 (Horizontal penalty factor)
        self.forward_encouragement_factor = forward_encouragement_factor # 前向鼓励因子 (Forward encouragement factor)
        self.heading_penalty_factor = heading_penalty_factor # 朝向惩罚因子 (Heading penalty factor)

        if(follow_trajectories): # 如果跟随轨迹 (If following trajectories)
            if(self.which_agent==1): # 如果是Ant智能体 (If agent is Ant)
                reward_func= self.ant_follow_traj # 设置奖励函数为ant_follow_traj (Set reward function to ant_follow_traj)
            elif(self.which_agent==2): # 如果是Swimmer智能体 (If agent is Swimmer)
                reward_func= self.swimmer_follow_traj # 设置奖励函数为swimmer_follow_traj (Set reward function to swimmer_follow_traj)
            elif(self.which_agent==4): # 如果是Cheetah智能体 (If agent is Cheetah)
                reward_func= self.cheetah_follow_traj # 设置奖励函数为cheetah_follow_traj (Set reward function to cheetah_follow_traj)
            else: # 其他智能体 (For other agents)
                # 为其他智能体定义或引发错误 (Define for other agents or raise an error)
                raise ValueError(f"Trajectory following reward not defined for agent: {self.which_agent}")
        else: # 如果不跟随轨迹 (If not following trajectories)
            if(self.which_agent==1): # 如果是Ant智能体 (If agent is Ant)
                reward_func= self.ant_forward # 设置奖励函数为ant_forward (Set reward function to ant_forward)
            elif(self.which_agent==2): # 如果是Swimmer智能体 (If agent is Swimmer)
                reward_func= self.swimmer_forward # 设置奖励函数为swimmer_forward (Set reward function to swimmer_forward)
            elif(self.which_agent==4): # 如果是Cheetah智能体 (If agent is Cheetah)
                reward_func= self.cheetah_forward # 设置奖励函数为cheetah_forward (Set reward function to cheetah_forward)
            elif(self.which_agent==6): # 如果是Hopper智能体 (If agent is Hopper)
                reward_func= self.hopper_forward # 设置奖励函数为hopper_forward (Set reward function to hopper_forward)
            else: # 其他智能体 (For other agents)
                 # 为其他智能体定义或引发错误 (Define for other agents or raise an error)
                raise ValueError(f"Forward reward not defined for agent: {self.which_agent}")
        return reward_func # 返回选择的奖励函数 (Return the selected reward function)

######################################################################################################################
    def ant_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                        curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Ant智能体跟随轨迹的奖励函数 (Reward function for Ant agent following trajectory)

        # 惩罚离轨迹的水平距离 (penalize horiz dist away from trajectory)
        scores[min_perp_dist<1] += (min_perp_dist*self.horiz_penalty_factor)[min_perp_dist<1] # 如果距离小于1 (If distance is less than 1)
        scores[min_perp_dist>=1] += (min_perp_dist*10*self.horiz_penalty_factor)[min_perp_dist>=1] # 如果距离大于等于1 (If distance is greater than or equal to 1)

        # 鼓励向前移动 (encourage moving-forward)
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0] # 如果未移动到下一段 (If not moved to next segment)
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1] # 如果移动到下一段 (If moved to next segment)

        # 防止高度过高或过低 (prevent height from going too high or too low)
        scores[pt[:,self.z_index]>0.67] += (self.heading_penalty_factor*40 + 0*pt[:,self.z_index])[pt[:,self.z_index]>0.67] # 如果高度大于0.67 (If height is greater than 0.67)
        scores[pt[:,self.z_index]<0.3] += (self.heading_penalty_factor*40 + 0*pt[:,self.z_index])[pt[:,self.z_index]<0.3] # 如果高度小于0.3 (If height is less than 0.3)

        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

    def swimmer_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                            curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Swimmer智能体跟随轨迹的奖励函数 (Reward function for Swimmer agent following trajectory)

        # 惩罚离轨迹的水平距离 (penalize horiz dist away from trajectory)
        scores += min_perp_dist*self.horiz_penalty_factor # 累加水平距离惩罚 (Add horizontal distance penalty)

        # 鼓励向前移动并惩罚不向前移动 (encourage moving-forward and penalize not-moving-forward)
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0] # 如果未移动到下一段 (If not moved to next segment)
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1] # 如果移动到下一段 (If moved to next segment)

        # （期望轨迹）线段与水平方向的夹角 (angle that (desired traj) line segment makes WRT horizontal)
        curr_line_start = self.desired_states[curr_seg] # 当前线段起点 (Current line segment start)
        curr_line_end = self.desired_states[curr_seg+1] # 当前线段终点 (Current line segment end)
        angle = np.arctan2(curr_line_end[:,1]-curr_line_start[:,1], curr_line_end[:,0]-curr_line_start[:,0]) # 计算角度 (-pi 到 pi) (Calculate angle (-pi to pi))
            # ^ -pi to pi

        # 惩罚偏离该角度的朝向 (penalize heading away from that angle)
        diff = np.abs(pt[:,self.yaw_index]-angle) # 计算角度差绝对值 (Calculate absolute difference in angle)
        diff[diff>np.pi]=(2*np.pi-diff)[diff>np.pi] # 如果计算的是绕圆的长路，则取较短的值作为差值 (if the calculation takes you the long way around the circle, take the shorter value instead as the difference)
            #^ if the calculation takes you the long way around the circle, 
            #take the shorter value instead as the difference
        my_range = np.pi/3.0 # 可接受的角度范围 (Acceptable angle range)
        scores[diff<my_range] += (self.heading_penalty_factor*diff)[diff<my_range] # 如果在范围内，按比例惩罚 (If within range, penalize proportionally)
        scores[diff>=my_range] += 20 # 如果超出范围，较大惩罚 (If out of range, larger penalty)

        # 不要弯曲过多 (dont bend in too much)
        first_joint = np.abs(pt[:,self.joint1_index]) # 第一个关节的角度绝对值 (Absolute angle of the first joint)
        second_joint = np.abs(pt[:,self.joint2_index]) # 第二个关节的角度绝对值 (Absolute angle of the second joint)
        limit = np.pi/3 # 弯曲限制 (Bending limit)
        scores[limit<first_joint] += 2 # 如果第一个关节超出限制 (If first joint exceeds limit)
        scores[limit<second_joint] += 2 # 如果第二个关节超出限制 (If second joint exceeds limit)

        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

    def cheetah_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                            curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Cheetah智能体跟随轨迹的奖励函数 (Reward function for Cheetah agent following trajectory)

        # 惩罚离轨迹的水平距离 (penalize horiz dist away from trajectory)
        scores += min_perp_dist*self.horiz_penalty_factor # 累加水平距离惩罚 (Add horizontal distance penalty)

        # 鼓励向前移动 (encourage moving-forward)
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0] # 如果未移动到下一段 (If not moved to next segment)
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1] # 如果移动到下一段 (If moved to next segment)

        # 不要将前胫向后移动太多以至于向前倾斜 (dont move front shin back so far that you tilt forward)
        front_leg = pt[:,self.frontleg_index] # 前腿位置 (Front leg position)
        my_range = 0.2 # 可接受范围 (Acceptable range)
        scores[front_leg>=my_range] += self.heading_penalty_factor # 如果超出范围，添加惩罚 (If out of range, add penalty)

        front_shin = pt[:,self.frontshin_index] # 前胫位置 (Front shin position)
        my_range = 0 # 可接受范围 (Acceptable range)
        scores[front_shin>=my_range] += self.heading_penalty_factor # 如果超出范围，添加惩罚 (If out of range, add penalty)

        front_foot = pt[:,self.frontfoot_index] # 前脚位置 (Front foot position)
        my_range = 0 # 可接受范围 (Acceptable range)
        scores[front_foot>=my_range] += self.heading_penalty_factor # 如果超出范围，添加惩罚 (If out of range, add penalty)

        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

######################################################################################################################
    def ant_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Ant智能体前向移动的奖励函数 (Reward function for Ant agent moving forward)

        # 注意高度 (watch the height)
        done_forever[pt[:,self.z_index] > 1] = 1 # 如果高度大于1，则标记为完成 (If height is greater than 1, mark as done)
        done_forever[pt[:,self.z_index] < 0.3] = 1 # 如果高度小于0.3，则标记为完成 (If height is less than 0.3, mark as done)

        # 动作惩罚 (action penalty)
        scaling= 150.0 # 缩放因子 (Scaling factor)
        if(pt_number==all_samples.shape[1]): # 如果是最后一个动作样本 (If it's the last action sample)
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number-1,:][done_forever==0]/scaling), axis=1) # 计算动作惩罚 (Calculate action penalty)
        else: # 否则 (Otherwise)
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number,:][done_forever==0]/scaling), axis=1) # 计算动作惩罚 (Calculate action penalty)

        # 速度奖励 (velocity reward)
        scores[done_forever==0] -= pt[:,self.xvel_index][done_forever==0] # 减去x方向速度作为奖励 (Subtract x-velocity as reward)

        # 生存奖励 (survival reward)
        scores[done_forever==0] -= 0.5 # 以前是0.05 (used to be 0.05)
        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

    def swimmer_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                        curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Swimmer智能体前向移动的奖励函数 (Reward function for Swimmer agent moving forward)

        ########### GYM ########### (Gym环境的奖励逻辑 - 已注释掉) (Reward logic for Gym environment - commented out)

        '''if(pt_number==all_samples.shape[1]):
            reward_ctrl = 0.0001 * np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1)
        else:
            reward_ctrl = 0.0001 * np.sum(np.square(all_samples[:,pt_number,:]), axis=1)
        reward_fwd = (pt[:,self.x_index]-prev_pt[:,self.x_index]) / 0.01'''

        ########### RLLAB ########### (rllab环境的奖励逻辑) (Reward logic for rllab environment)

        scaling=50.0 # 缩放因子 (Scaling factor)
        if(pt_number==all_samples.shape[1]): # 如果是最后一个动作样本 (If it's the last action sample)
            reward_ctrl = 0.5 * np.sum(np.square(all_samples[:,pt_number-1,:]/scaling), axis=1) # 计算控制奖励 (Calculate control reward)
        else: # 否则 (Otherwise)
            reward_ctrl = 0.5 * np.sum(np.square(all_samples[:,pt_number,:]/scaling), axis=1) # 计算控制奖励 (Calculate control reward)
        reward_fwd = pt[:,self.xvel_index] # 前向奖励为x方向速度 (Forward reward is x-velocity)

        #########################

        scores += -reward_fwd + reward_ctrl # 更新分数 (Update scores)
        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

    def cheetah_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward,
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Cheetah智能体前向移动的奖励函数 (Reward function for Cheetah agent moving forward)

        ########### GYM ########### (Gym环境的奖励逻辑 - 已注释掉) (Reward logic for Gym environment - commented out)

        '''#action
        if(pt_number==all_samples.shape[1]):
            scores += 0.1*np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1)
        else:
            scores += 0.1*np.sum(np.square(all_samples[:,pt_number,:]), axis=1)

        #velocity
        scores -= (pt[:,self.x_index]-prev_pt[:,self.x_index]) / 0.01'''

        ########### RLLAB ########### (rllab环境的奖励逻辑) (Reward logic for rllab environment)

        # 动作惩罚 (action penalty)
        if(pt_number==all_samples.shape[1]): # 如果是最后一个动作样本 (If it's the last action sample)
            scores += 0.05*np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1) # 计算动作惩罚 (Calculate action penalty)
        else: # 否则 (Otherwise)
            scores += 0.05*np.sum(np.square(all_samples[:,pt_number,:]), axis=1) # 计算动作惩罚 (Calculate action penalty)

        # 速度奖励 (velocity reward)
        scores -= pt[:,self.xvel_index] # 减去x方向速度作为奖励 (Subtract x-velocity as reward)

        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)

    def hopper_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number): # Hopper智能体前向移动的奖励函数 (Reward function for Hopper agent moving forward)

        scaling=200.0 # 缩放因子 (Scaling factor)

        # 不要让方向超出范围 (dont tilt orientation out of range)
        orientation = pt[:,self.orientation_index] # 获取方向 (Get orientation)
        done_forever[np.abs(orientation)>= 0.3] = 1 # 如果方向绝对值大于等于0.3，标记为完成 (If absolute orientation is greater than or equal to 0.3, mark as done)

        # 不要摔倒到地面 (dont fall to ground)
        done_forever[pt[:,self.z_index] <= 0.7] = 1 # 如果z坐标小于等于0.7，标记为完成 (If z-coordinate is less than or equal to 0.7, mark as done)

        # 动作惩罚 (action penalty)
        if(pt_number==all_samples.shape[1]): # 如果是最后一个动作样本 (If it's the last action sample)
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number-1,:][done_forever==0]/scaling), axis=1) # 计算动作惩罚 (Calculate action penalty)
        else: # 否则 (Otherwise)
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number,:][done_forever==0])/scaling, axis=1) # 计算动作惩罚 (Calculate action penalty)

        # 速度奖励 (velocity reward)
        scores[done_forever==0] -= pt[:,self.xvel_index][done_forever==0] # 减去x方向速度作为奖励 (Subtract x-velocity as reward)

        # 生存奖励 (survival reward)
        scores[done_forever==0] -= 1 # 生存奖励为-1 (Survival reward is -1)

        return scores, done_forever # 返回分数和完成标志 (Return scores and done flag)