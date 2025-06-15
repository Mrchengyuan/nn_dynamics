
import torch # 导入torch库
import torch.nn as nn # 导入torch.nn模块，用于构建神经网络

class FeedforwardNetwork(nn.Module): # 定义一个名为FeedforwardNetwork的类，继承自nn.Module
    def __init__(self, inputSize, outputSize, num_fc_layers, depth_fc_layers): # 定义构造函数
        super(FeedforwardNetwork, self).__init__() # 调用父类的构造函数

        self.layers = nn.ModuleList() # 创建一个ModuleList来存储网络层

        current_size = inputSize # 设置当前层的大小为输入大小
        for i in range(num_fc_layers): # 循环创建全连接层
            self.layers.append(nn.Linear(current_size, depth_fc_layers)) # 添加一个线性层（全连接层）
            self.layers.append(nn.ReLU()) # 添加一个ReLU激活函数层
            current_size = depth_fc_layers # 更新当前层的大小为隐藏层深度

        self.layers.append(nn.Linear(current_size, outputSize)) # 添加最后一个线性层（输出层）
        # PyTorch的nn.Linear默认使用Xavier/Glorot初始化，与tf.contrib.layers.xavier_initializer(uniform=False)类似

    def forward(self, x): # 定义前向传播函数
        for layer in self.layers: # 遍历所有层
            x = layer(x) # 将输入x传递给当前层
        return x # 返回最终的输出