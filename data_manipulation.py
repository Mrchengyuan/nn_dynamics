import numpy as np # 导入numpy库，用于数值计算 (Import numpy for numerical computation)
import numpy.random as npr # 导入numpy的随机数生成模块 (Import numpy's random number generation module)
# import tensorflow as tf # TensorFlow导入已移除，因为未使用 (TensorFlow import removed as it's not used)
import time # 导入time模块，用于时间相关操作 (Import time module for time-related operations)
import math # 导入math模块，用于数学运算 (Import math module for mathematical operations)
import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图 (Import matplotlib's pyplot module for plotting)
import copy # 导入copy模块，用于创建对象的副本 (Import copy module for creating copies of objects)

def get_indices(which_agent): # 定义获取索引的函数 (Define function to get indices)
    x_index = -7 # 初始化x索引 (Initialize x index)
    y_index = -7 # 初始化y索引 (Initialize y index)
    z_index = -7  # 初始化z索引 (Initialize z index)
    yaw_index = -7 # 初始化偏航角索引 (Initialize yaw index)
    joint1_index = -7  # 初始化关节1索引 (Initialize joint1 index)
    joint2_index = -7  # 初始化关节2索引 (Initialize joint2 index)
    frontleg_index = -7 # 初始化前腿索引 (Initialize frontleg index)
    frontshin_index = -7 # 初始化前胫索引 (Initialize frontshin index)
    frontfoot_index = -7 # 初始化前脚索引 (Initialize frontfoot index)
    xvel_index = -7 # 初始化x速度索引 (Initialize xvel index)
    orientation_index = -7 # 初始化方向索引 (Initialize orientation index)

    if(which_agent==0): # 如果是PointMass智能体 (If agent is PointMass)
        x_index= 0 # 设置x索引 (Set x index)
        y_index= 1 # 设置y索引 (Set y index)
    elif(which_agent==1): # 如果是Ant智能体 (If agent is Ant)
        x_index= 29 # 设置x索引 (Set x index)
        y_index= 30 # 设置y索引 (Set y index)
        z_index = 31 # 设置z索引 (Set z index)
        xvel_index = 38 # 设置x速度索引 (Set xvel index)
    elif(which_agent==2): # 如果是Swimmer智能体 (If agent is Swimmer)
        x_index= 10 # 设置x索引 (Set x index)
        y_index= 11 # 设置y索引 (Set y index)
        yaw_index = 2 # 设置偏航角索引 (Set yaw index)
        joint1_index = 3 # 设置关节1索引 (Set joint1 index)
        joint2_index = 4 # 设置关节2索引 (Set joint2 index)
        xvel_index = 13 # 设置x速度索引 (Set xvel index)
    elif(which_agent==3): # 如果是Reacher智能体 (If agent is Reacher)
        x_index= 6 # 设置x索引 (Set x index)
        y_index= 7 # 设置y索引 (Set y index)
    elif(which_agent==4): # 如果是Cheetah智能体 (If agent is Cheetah)
        x_index= 18 # 设置x索引 (Set x index)
        y_index= 20 # 设置y索引 (Set y index)
        frontleg_index = 6 # 设置前腿索引 (Set frontleg index)
        frontshin_index = 7 # 设置前胫索引 (Set frontshin index)
        frontfoot_index = 8 # 设置前脚索引 (Set frontfoot index)
        xvel_index = 21 # 设置x速度索引 (Set xvel index)
    elif(which_agent==5): # 如果是Roach智能体 (非MuJoCo) (If agent is Roach (not MuJoCo))
        x_index= 0 # 设置x索引 (Set x index)
        y_index= 1 # 设置y索引 (Set y index)
    elif(which_agent==6): # 如果是Hopper智能体 (If agent is Hopper)
        x_index = 11 # 设置x索引 (Set x index)
        y_index = 13 # 设置y索引 (Set y index)
        z_index = 0 # 设置z索引 (Set z index)
        xvel_index = 14 # 设置x速度索引 (Set xvel index)
        orientation_index = 1 # 设置方向索引 (Set orientation index)
    elif(which_agent==7): # 如果是Walker智能体 (If agent is Walker)
        x_index = 18 # 设置x索引 (Set x index)
        y_index = 20 # 设置y索引 (Set y index)

    return x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, \
            frontshin_index, frontfoot_index, xvel_index, orientation_index # 返回所有索引 (Return all indices)

def generate_training_data_inputs(states0, controls0): # 定义生成训练数据输入的函数 (Define function to generate training data inputs)
    # 初始化变量 (init vars)
    states=np.copy(states0) # 复制状态 (Copy states)
    controls=np.copy(controls0) # 复制控制 (Copy controls)
    new_states=[] # 初始化新状态列表 (Initialize list for new states)
    new_controls=[] # 初始化新控制列表 (Initialize list for new controls)

    # 移除每个rollout中的最后一个条目（因为它没有关联的“输出”）(remove the last entry in each rollout (because that entry doesn't have an associated "output"))
    for i in range(len(states)): # 对于每个rollout (For each rollout)
        curr_item = states[i] # 获取当前状态序列 (Get current state sequence)
        length = curr_item.shape[0] # 获取长度 (Get length)
        new_states.append(curr_item[0:length-1,:]) # 添加除最后一个之外的所有状态 (Add all states except the last one)

        curr_item = controls[i] # 获取当前控制序列 (Get current control sequence)
        length = curr_item.shape[0] # 获取长度 (Get length)
        new_controls.append(curr_item[0:length-1,:]) # 添加除最后一个之外的所有控制 (Add all controls except the last one)
   
    # 将rollout列表转换为一个大的数据数组 (turn the list of rollouts into just one large array of data)
    dataX= np.concatenate(new_states, axis=0) # 连接所有新状态 (Concatenate all new states)
    dataY= np.concatenate(new_controls, axis=0) # 连接所有新控制 (Concatenate all new controls)
    return dataX, dataY # 返回dataX和dataY (Return dataX and dataY)

def generate_training_data_outputs(states, which_agent): # 定义生成训练数据输出的函数 (Define function to generate training data outputs)
    # 对于每个rollout，对应于每个(s_i)的输出是(s_i+1 - s_i) (for each rollout, the output corresponding to each (s_i) is (s_i+1 - s_i))
    differences=[] # 初始化差异列表 (Initialize list for differences)
    for states_in_single_rollout in states: # 对于单个rollout中的状态 (For states in a single rollout)
        output = states_in_single_rollout[1:states_in_single_rollout.shape[0],:] \
                -states_in_single_rollout[0:states_in_single_rollout.shape[0]-1,:] # 计算状态差异 (Calculate state differences)
        differences.append(output) # 添加差异到列表 (Add differences to list)
    output = np.concatenate(differences, axis=0) # 连接所有差异 (Concatenate all differences)
    return output # 返回输出 (Return output)

def from_observation_to_usablestate(states, which_agent, just_one): # 定义将观测转换为可用状态的函数 (Define function to convert observation to usable state)

    #######################################
    ######### POINTMASS ###################
    #######################################

    #0: x (x坐标)
    #1: y (y坐标)
    #2: vx (x方向速度)
    #3: vy (y方向速度)
    if(which_agent==0): # 如果是PointMass智能体 (If agent is PointMass)
        return states # 直接返回状态 (Return states directly)

    #######################################
    ######### ANT #########################
    #######################################

    # 我们使用以下观测作为NN的输入（41个值）(we use the following observation as input to NN (41 things))
        #0到14... 15个关节位置 (0 to 14... 15 joint positions)
        #15到28... 14个关节速度 (15 to 28... 14 joint velocities)
        #29到31... 3个身体质心位置 (29 to 31... 3 body com pos)
        #32到37... 3个身体角度的6个cos和sin值（来自9个旋转矩阵值）(32 to 37... 6 cos and sin of 3 body angles (from 9 rotation mat))
        #38到40... 身体质心速度 (38 to 40... body com vel)

    # env.step返回的值 (returned by env.step)
        #0到14 = 位置 (0 to 14 = positions)
            #j0 x位置 (j0 x position)
            #j1 y位置 (j1 y position)
            #j2 z位置 (j2 z position)
            #3 ?
            #4 5 身体翻转 (4 5 body flip)
            #6 身体旋转 (6 body rotate)
            #7 腿部偏航（逆时针），8 腿部向下弯曲 (7 leg yaw ccw, 8 leg bend down)
            #9, 10
            #11, 12
            #13,14 
        #15到28 = 速度 (15 to 28 = velocities)
        #29到37 = 旋转矩阵（9个值）(29 to 37 = rotation matrix (9))
        #38到40 = 质心位置 (38 to 40 = com positions)
        #41到43 = 质心速度 (41 to 43 = com velocities)

    if(which_agent==1): # 如果是Ant智能体 (If agent is Ant)
        if(just_one): # 如果只是单个状态 (If just one state)
            curr_item = np.copy(states) # 复制状态 (Copy states)
            joint_pos = curr_item[0:15] # 关节位置 (Joint positions)
            joint_vel = curr_item[15:29] # 关节速度 (Joint velocities)
            body_pos = curr_item[38:41] # 身体位置 (Body position)
            body_rpy = to_euler(curr_item[29:38], just_one) # 身体姿态（欧拉角的cos/sin）(Body rotation (cos/sin of Euler angles))
            body_vel = curr_item[41:44] # 身体速度 (Body velocity)
            full_item = np.concatenate((joint_pos, joint_vel, body_pos, body_rpy, body_vel), axis=0) # 连接所有部分 (Concatenate all parts)
            return full_item # 返回完整项目 (Return full item)

        else: # 如果是多个状态序列 (If multiple state sequences)
            new_states=[] # 初始化新状态列表 (Initialize list for new states)
            for i in range(len(states)): # 对于每个rollout (for each rollout)
                curr_item = np.copy(states[i]) # 复制当前rollout的状态 (Copy states of current rollout)

                joint_pos = curr_item[:,0:15] # 关节位置 (Joint positions)
                joint_vel = curr_item[:,15:29] # 关节速度 (Joint velocities)
                body_pos = curr_item[:,38:41] # 身体位置 (Body position)
                body_rpy = to_euler(curr_item[:,29:38], just_one) # 身体姿态 (Body rotation)
                body_vel = curr_item[:,41:44] # 身体速度 (Body velocity)
                
                full_item = np.concatenate((joint_pos, joint_vel, body_pos, body_rpy, body_vel), axis=1) # 连接所有部分 (Concatenate all parts)
                new_states.append(full_item) # 添加到新状态列表 (Add to new states list)
            return new_states # 返回新状态列表 (Return new states list)


    #######################################
    ######### SWIMMER #####################
    #######################################

    # 总共16个值 (total = 16)
        #0 slider x... 1 slider y.... 2 朝向 (0 slider x... 1 slider y.... 2 heading)
        #3,4 两个铰链关节位置 (3,4 the two hinge joint pos)
        #5,6 slider x/y 速度 (5,6 slider x/y vel)
        #7 朝向速度 (7 heading vel)
        #8,9 两个铰链关节速度 (8,9 the two hinge joint vel)
        #10,11,12 质心x,y,z位置 (10,11,12 cm x and y and z pos)
        #13,14,15 质心x,y,z速度 (13,14,15 cm x and y and z vel)
    if(which_agent==2): # 如果是Swimmer智能体 (If agent is Swimmer)
        return states # 直接返回状态 (Return states directly)

    #######################################
    ######### REACHER #####################
    #######################################

    # 总共11个值 (total = 11)
        # 2-- 两个角度的cos(theta) (2-- cos(theta) of the 2 angles)
        # 2-- 两个角度的sin(theta) (2-- sin(theta) of the 2 angles)
        # 2-- 目标位置-------------------（忽略此项）(2-- goal pos -------------------(ignore this))
        # 2-- 两个角度的速度 (2-- vel of the 2 angles)
        # 3-- 指尖质心 (3-- fingertip cm)
    if(which_agent==3): # 如果是Reacher智能体 (If agent is Reacher)
        if(just_one): # 如果只是单个状态 (If just one state)
            curr_item = np.copy(states) # 复制状态 (Copy states)
            keep_1 = curr_item[0:4] # 保留部分1 (Keep part 1)
            keep_2 = curr_item[6:11] # 保留部分2 (Keep part 2)
            full_item = np.concatenate((keep_1, keep_2), axis=0) # 连接保留的部分 (Concatenate kept parts)
            return full_item # 返回完整项目 (Return full item)

        else: # 如果是多个状态序列 (If multiple state sequences)
            new_states=[] # 初始化新状态列表 (Initialize list for new states)
            for i in range(len(states)): # 对于每个rollout (for each rollout)
                curr_item = np.copy(states[i]) # 复制当前rollout的状态 (Copy states of current rollout)
                keep1 = curr_item[:,0:4] # 保留部分1 (Keep part 1)
                keep2 = curr_item[:,6:11] # 保留部分2 (Keep part 2)
                full_item = np.concatenate((keep1, keep2), axis=1) # 连接保留的部分 (Concatenate kept parts)
                new_states.append(full_item) # 添加到新状态列表 (Add to new states list)
            return new_states # 返回新状态列表 (Return new states list)

    #######################################
    ######### HALF CHEETAH ################
    #######################################

    # 当你传递某些东西来重置环境时的状态（33个值）(STATE when you pass in something to reset env: (33))
    #    rootx, rootz, rooty (根节点x, z, y坐标)
    #    bthigh, bshin, bfoot (后大腿、小腿、脚)
    #    fthigh, fshin, ffoot (前大腿、小腿、脚)
    #    rootx, rootz, rooty --vel (根节点x, z, y方向速度)
    #    bthigh, bshin, bfoot --vel (后大腿、小腿、脚的速度)
    #    fthigh, fshin, ffoot --vel (前大腿、小腿、脚的速度)
    # self.model.data.qacc (9) (关节加速度)
    # self.model.data.ctrl (6) (控制信号)
    # 观测（24个值）(OBSERVATION: (24))
    #    0: rootx (前进/后退) (0: rootx (forward/backward))
    #    1: rootz (上/下) (1: rootz (up/down))
    #    2: rooty (身体角度) (2: rooty (angle of body))
    #    3: bthigh (+ 表示向后移动) (3: bthigh (+ is move back))
    #    4: bshin (后小腿)
    #    5: bfoot (后脚)
    #    6: fthigh (前大腿)
    #    7: fshin (前小腿)
    #    8: ffoot (前脚)
    #    9: root x vel (根节点x方向速度)
    #    10: root z vel (根节点z方向速度)
    #    11: root y vel (根节点y方向速度)
    #    12: bthigh vel (后大腿速度)
    #    13: bshin vel (后小腿速度)
    #    14: bfoot vel (后脚速度)
    #    15: fthigh vel (前大腿速度)
    #    16: fshin vel (前小腿速度)
    #    17: ffoot vel (前脚速度)
    #com x (质心x)
    #com y (质心y)
    #com z (质心z)
    #com vx (质心vx)
    #com vy (质心vy)
    #com vz (质心vz)

    if(which_agent==4): # 如果是Cheetah智能体 (If agent is Cheetah)
        return states # 直接返回状态 (Return states directly)

    #######################################
    ######### ROACH (personal env) ########
    #######################################

        # x,y,z com position (质心x,y,z位置)
        # orientation com (质心方向)
        # cos of 2 motor positions (2个电机位置的cos值)
        # sin of 2 motor positions (2个电机位置的sin值)
        # com velocity (质心速度)
        # orientation angular vel (方向角速度)
        # 2 motor vel (2个电机速度)
    
    elif(which_agent==5): # 如果是Roach智能体 (If agent is Roach)
        if(just_one): # 如果只是单个状态 (If just one state)
            curr_item = np.copy(states) # 复制状态 (Copy states)
            keep_1 = curr_item[0:6] # 保留部分1 (Keep part 1)
            two = np.cos(curr_item[6:8]) # 计算cos值 (Calculate cos values)
            three = np.sin(curr_item[6:8]) # 计算sin值 (Calculate sin values)
            keep_4 = curr_item[8:16] # 保留部分4 (Keep part 4)
            full_item = np.concatenate((keep_1, two, three, keep_4), axis=0) # 连接所有部分 (Concatenate all parts)
            return full_item # 返回完整项目 (Return full item)

        else: # 如果是多个状态序列 (If multiple state sequences)
            new_states=[] # 初始化新状态列表 (Initialize list for new states)
            for i in range(len(states)): # 对于每个rollout (for each rollout)
                curr_item = np.copy(states[i]) # 复制当前rollout的状态 (Copy states of current rollout)
                keep1 = curr_item[:,0:6] # 保留部分1 (Keep part 1)
                two = np.cos(curr_item[:,6:8]) # 计算cos值 (Calculate cos values)
                three = np.sin(curr_item[:,6:8]) # 计算sin值 (Calculate sin values)
                keep4 = curr_item[:,8:16] # 保留部分4 (Keep part 4)
                full_item = np.concatenate((keep1, two, three, keep4), axis=1) # 连接所有部分 (Concatenate all parts)
                new_states.append(full_item) # 添加到新状态列表 (Add to new states list)
            return new_states # 返回新状态列表 (Return new states list)

    #######################################
    ######### HOPPER ######################
    #######################################

    # 观测：17个值 (observation: 17 things)
        #5个关节 -- j0 (高度), j2, j3, j4, j5 (5 joints-- j0 (height), j2, j3, j4, j5)
        #6个速度 (6 velocities)
        #3个质心位置 (3 com pos)
        #3个质心速度 (3 com vel)
    # 状态：21个值 (state: 21 things)
        #6个关节位置 (6 joint pos)
        #6个关节速度 (6 joint vel)
        #6个关节加速度 (6 qacc)
        #3个控制信号 (3 ctrl)

    if(which_agent==6): # 如果是Hopper智能体 (If agent is Hopper)
        return states # 直接返回状态 (Return states directly)

    #######################################
    ######### WALKER ######################
    #######################################
    
    # 观测：24个值 (observation: 24 things)
        #9个关节位置 (9 joint pos)
        #9个速度 (9 velocities)
        #3个质心位置 (3 com pos)
        #3个质心速度 (3 com vel)

    if(which_agent==7): # 如果是Walker智能体 (If agent is Walker)
        return states # 直接返回状态 (Return states directly)


def to_euler(rot_mat, just_one): # 定义将旋转矩阵转换为欧拉角（的cos/sin）的函数 (Define function to convert rotation matrix to Euler angles (cos/sin))
    if(just_one): # 如果只是单个旋转矩阵 (If just one rotation matrix)
        # 计算r, p, y欧拉角 (Calculate r, p, y Euler angles)
        # 注意：这里的arctan2参数顺序和具体哪个轴对应哪个角度可能需要根据具体旋转矩阵定义来确认 (Note: The order of arctan2 parameters and which axis corresponds to which angle might need confirmation based on the specific rotation matrix definition)
        r=np.arctan2(rot_mat[3], rot_mat[1]) # 计算r角 (Calculate r angle)
        p=np.arctan2(-rot_mat[6], np.sqrt(rot_mat[7]*rot_mat[7]+rot_mat[8]*rot_mat[8])) # 计算p角 (Calculate p angle)
        y=np.arctan2(rot_mat[7], rot_mat[8]) # 计算y角 (Calculate y angle)

        return np.array([np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)]) # 返回欧拉角的cos和sin值 (Return cos and sin values of Euler angles)

    else: # 如果是旋转矩阵序列 (If a sequence of rotation matrices)
        r=np.arctan2(rot_mat[:,3], rot_mat[:,1]) # 计算r角序列 (Calculate r angle sequence)
        r=np.concatenate((np.expand_dims(np.cos(r), axis=1), np.expand_dims(np.sin(r), axis=1)), axis=1) # 连接r角的cos和sin值 (Concatenate cos and sin values of r angle)

        p=np.arctan2(-rot_mat[:,6], np.sqrt(rot_mat[:,7]*rot_mat[:,7]+rot_mat[:,8]*rot_mat[:,8])) # 计算p角序列 (Calculate p angle sequence)
        p=np.concatenate((np.expand_dims(np.cos(p), axis=1), np.expand_dims(np.sin(p), axis=1)), axis=1) # 连接p角的cos和sin值 (Concatenate cos and sin values of p angle)

        y=np.arctan2(rot_mat[:,7], rot_mat[:,8]) # 计算y角序列 (Calculate y angle sequence)
        y=np.concatenate((np.expand_dims(np.cos(y), axis=1), np.expand_dims(np.sin(y), axis=1)), axis=1) # 连接y角的cos和sin值 (Concatenate cos and sin values of y angle)

        return np.concatenate((r,p,y), axis=1) # 返回连接后的欧拉角cos和sin值序列 (Return concatenated sequence of Euler angle cos and sin values)
