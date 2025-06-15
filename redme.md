# 在 Mac Mini M2 上安装 MuJoCo 并运行本项目

以下步骤适用于 macOS (Apple Silicon) 环境，假设已安装 Python 3.10。

1. **安装 Python 依赖**

```bash
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip3 install gymnasium[mujoco] matplotlib pyyaml
```

2. **安装 MuJoCo**

- 访问 [MuJoCo Releases](https://github.com/google-deepmind/mujoco/releases) 下载适用于 macOS 的 DMG。
- 打开 DMG，将 `MuJoCo.app` 拖入 `Applications` 目录即可。
- Python 绑定通过 `pip install mujoco` 已随 `gymnasium[mujoco]` 安装，无需额外配置。

3. **运行示例**

克隆代码仓库并在项目根目录执行：

```bash
python3 main.py --seed=0 --run_num=0 --yaml_file=swimmer_forward
```

该命令将在 `run_0` 目录下生成训练数据并训练简单的动力学模型。
如果想在采样或MPC回放过程中看到实时画面，可以在命令中加入 `--might_render` 或 `--visualize_MPC_rollout` 参数，环境将以 Gym 的人类渲染模式显示。

