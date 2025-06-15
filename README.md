# Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning on Apple Silicon (M2 Mac Mini)

This document provides instructions to set up the environment for the project "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning" on a Mac Mini M2 (Apple Silicon).

**Original Project ArXiv Link**: [https://arxiv.org/abs/1708.02596](https://arxiv.org/abs/1708.02596)

## Environment Setup for Mac Mini M2

This guide focuses on setting up Python 3.10, PyTorch 1.12.1, and the appropriate MuJoCo version compatible with Apple Silicon.

### 1. Python 3.10 Installation

It is recommended to manage Python versions using `pyenv` or `conda`.

**Using pyenv (Recommended for macOS):**

1.  **Install Homebrew** (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    Follow the instructions to add Homebrew to your PATH.

2.  **Install pyenv and build dependencies:**
    ```bash
    brew update
    brew install pyenv openssl readline sqlite3 xz zlib tcl-tk
    ```
    For macOS Mojave (10.14) and higher, you might need to install SDK headers:
    ```bash
    sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target / # Adjust version if necessary
    ```

3.  **Configure pyenv in your shell:**
    Add the following lines to your shell configuration file (e.g., `~/.zshrc` for Zsh, `~/.bashrc` or `~/.bash_profile` for Bash):
    ```bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv 1>/dev/null 2>&1; then
      eval "$(pyenv init --path)"
      eval "$(pyenv init -)"
    fi
    ```
    Restart your shell for changes to take effect:
    ```bash
    exec "$SHELL"
    ```

4.  **Install Python 3.10:**
    ```bash
    pyenv install 3.10.14 # Or the latest 3.10.x version available
    ```

5.  **Set Python 3.10.x as global or local:**
    To set it globally:
    ```bash
    pyenv global 3.10.14
    ```
    To set it for a specific project (in the project directory):
    ```bash
    pyenv local 3.10.14
    ```
    Verify the Python version:
    ```bash
    python --version # Or python3 --version
    ```

**Using Miniconda/Anaconda:**

1.  **Download and install Miniconda:**
    Choose the macOS installer (pkg or bash) for Apple Silicon (arm64 or universal) from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
    For example, using the bash installer:
    ```bash
    bash Miniconda3-latest-MacOSX-arm64.sh # Or the specific version you downloaded
    ```
    Follow the prompts. Allow the installer to initialize Conda.

2.  **Create and activate a new environment for Python 3.10:**
    ```bash
    conda create -n myenv python=3.10
    conda activate myenv
    ```
    Replace `myenv` with your preferred environment name.

### 2. PyTorch 1.12.1 Installation

PyTorch 1.12.1 can be installed via pip or conda. These versions are compatible with Python 3.10 and Apple Silicon, leveraging Metal Performance Shaders (MPS) for GPU acceleration.

**Using pip:**
```bash
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

**Using conda (if you installed Python via conda):**
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```

**Verify PyTorch installation and MPS support:**
Open a Python interpreter:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"MPS device is available. Test tensor on MPS: {x}")
else:
    print("MPS device is not available.")

# Example tensor operation
x = torch.rand(5, 3)
print(x)
```

### 3. MuJoCo Installation

MuJoCo is now open-source and does not require a license key.

1.  **Download MuJoCo Pre-built Libraries:**
    Go to the [MuJoCo GitHub Releases page](https://github.com/google-deepmind/mujoco/releases) and download the latest macOS (Apple Silicon / arm64) DMG file.

2.  **Install MuJoCo:**
    Open the DMG file. You will find `MuJoCo.app` and a `mujoco.framework` directory.
    *   You can run `MuJoCo.app` directly to test simulations (e.g., drag `model/humanoid/humanoid.xml` onto it).
    *   For system-wide availability, you can drag `MuJoCo.app` to your `/Applications` folder.
    *   The `mujoco.framework` is important for development if you are building projects against the C API directly. For Python bindings, this might not be explicitly needed if the Python package bundles the library.

3.  **Install MuJoCo Python Bindings:**
    The official Python bindings are recommended.
    ```bash
    pip3 install mujoco
    ```
    This package includes a copy of the MuJoCo library.

4.  **Environment Variables (Optional but Recommended):**
    While the `mujoco` pip package often handles paths, some tools or older projects (like those based on `mujoco-py`) might still look for MuJoCo via environment variables. It's good practice to set them, especially if you have a separate MuJoCo installation (e.g., from the DMG).
    Add these to your shell configuration file (`~/.zshrc` or `~/.bashrc`):
    ```bash
    # Adjust if you placed MuJoCo.app elsewhere or want to point to the framework
    export MUJOCO_PATH="$HOME/.mujoco/mujoco210" # Example path, adjust as needed if you manually copy framework/binaries.
                                                # If using pip installed mujoco primarily, this might not be strictly needed.
    # For mujoco-py (if used by older projects, less relevant for the official 'mujoco' package)
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
    # export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
    # export MUJOCO_PY_MJKEY_PATH="$HOME/.mujoco/mjkey.txt" # mjkey.txt is no longer needed for official MuJoCo releases
    ```
    **Note:** The `mujoco` pip package aims to be self-contained. The `MUJOCO_PATH` might be more relevant if you're also using the C libraries directly or specific tools that look for it. For the Python package `mujoco`, explicit `LD_LIBRARY_PATH` or `MUJOCO_PY_...` variables are generally not required as it bundles its own binaries.

5.  **Test MuJoCo Python installation:**
    Open a Python interpreter:
    ```python
    import mujoco
    import numpy as np

    # Example from MuJoCo documentation
    xml = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    mujoco.mj_step(model, data)
    print("Box position:", data.geom("red_box").xpos)
    print("Sphere position:", data.geom("green_sphere").xpos)
    ```

### 4. Install Core RL Library and Other Dependencies

This project uses `gymnasium` as the environment interface and `garage` for TRPO and related functionalities.

```bash
pip3 install gymnasium[mujoco] # Installs gymnasium and MuJoCo bindings
pip3 install garage             # RL library for TRPO and other algorithms
pip3 install cloudpickle        # For serialization
pip3 install seaborn            # For plotting
pip3 install matplotlib         # For plotting (if not already installed as a dependency of seaborn)
```
**Note:** The original project used `gym` and `rllab`. These have been updated to `gymnasium` and `garage` respectively. The `mujoco` PyPI package (installed in step 3) provides the MuJoCo physics engine. `gymnasium[mujoco]` ensures that Gymnasium environments for MuJoCo are installed and compatible.

### 5. A Note on Testing

This codebase has been significantly refactored to use PyTorch, Gymnasium, and Garage, replacing the original TensorFlow and rllab implementations. Due to limitations in the automated testing environment (sandbox errors preventing script execution), end-to-end script execution and full functionality (especially of the TRPO components via Garage) could not be verified. Users are advised to run tests in their own local setup. The DAgger component in `mbmf.py` also relies on `GetTrueAction` which has been converted but its full integration for label generation requires further testing.

### 6. Running the Code

The original `readme.md` contains instructions on how to run the experiments. These sections are preserved below. Ensure you are using `python3` if `python` defaults to Python 2 or is not found. If you used `pyenv` or `conda` to set up Python 3.10, `python` should correctly point to Python 3.10.

---
*(Original README sections will be appended here in the next step)*
---

## Original Project Information (from existing readme.md)

[Arxiv Link](https://arxiv.org/abs/1708.02596)

**Abstract**: Model-free deep reinforcement learning algorithms have been shown to be capable of learning a wide range of robotic skills, but typically require a very large number of samples to achieve good performance. Model-based algorithms, in principle, can provide for much more efficient learning, but have proven difficult to extend to expressive, high-capacity models such as deep neural networks. In this work, we demonstrate that medium-sized neural network models can in fact be combined with model predictive control (MPC) to achieve excellent sample complexity in a model-based reinforcement learning algorithm, producing stable and plausible gaits to accomplish various complex locomotion tasks. We also propose using deep neural network dynamics models to initialize a model-free learner, in order to combine the sample efficiency of model-based approaches with the high task-specific performance of model-free methods. We empirically demonstrate on MuJoCo locomotion tasks that our pure model-based approach trained on just minutes of random action data can follow arbitrary trajectories, and that our hybrid algorithm can accelerate model-free learning on high-speed benchmark tasks, achieving sample efficiency gains of 3-5x on swimmer, cheetah, hopper, and ant agents.

- For notes on how to use your own environment, how to edit envs, etc. go to [notes.md](docs/notes.md) (Note: This link might need to be updated if the repo structure changes or if it's a relative link in the original project).

---------------------------------------------------------------

### How to run everything

```bash
cd scripts
./swimmer_mbmf.sh
./cheetah_mbmf.sh
./hopper_mbmf.sh
./ant_mbmf.sh
```

Each of those scripts does something similar to the following (but for multiple seeds):

```bash
python3 main.py --seed=0 --run_num=1 --yaml_file='swimmer_forward'
python3 mbmf.py --run_num=1 --which_agent=2 --std_on_mlp_policy=1.0 # std_on_mlp_policy was in the original shell script
python3 trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=2 --std_on_mlp_policy=0.5 # --num_workers_trpo removed
python3 plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=0.5 --which_agent=2 --run_nums 1 --seeds 0
```
Note that `[trpo_dir]` above corresponds to where the TRPO runs are saved by Garage. This is typically in a subfolder under `data/local/experiment/` within your project directory (e.g., `data/local/experiment/trpo_run_mf_swimmer_seed0_exp`). Each of these steps are further explained in the following sections.
**Note:** `python` has been changed to `python3` assuming a modern setup. If your `python` command correctly points to Python 3.10, you can use `python`.

---------------------------------------------------------------

### How to run MB

Need to specify:

&nbsp;&nbsp;&nbsp;&nbsp;--**yaml_file** Specify the corresponding yaml file
&nbsp;&nbsp;&nbsp;&nbsp;--**seed** Set random seed to set for numpy and tensorflow
&nbsp;&nbsp;&nbsp;&nbsp;--**run_num** Specify what directory to save files under
&nbsp;&nbsp;&nbsp;&nbsp;--**use_existing_training_data** To use the data that already exists in the directory run_num instead of recollecting
&nbsp;&nbsp;&nbsp;&nbsp;--**desired_traj_type** What type of trajectory to follow (if you want to follow a trajectory)
&nbsp;&nbsp;&nbsp;&nbsp;--**num_rollouts_save_for_mf** Number of on-policy rollouts to save after last aggregation iteration, to be used later
&nbsp;&nbsp;&nbsp;&nbsp;--**might_render** If you might want to visualize anything during the run
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_MPC_rollout** To set a breakpoint and visualize the on-policy rollouts after each agg iteration
&nbsp;&nbsp;&nbsp;&nbsp;--**perform_forwardsim_for_vis** To visualize an open-loop prediction made by the learned dynamics model
&nbsp;&nbsp;&nbsp;&nbsp;--**print_minimal** To not print messages regarding progress/notes/etc.

##### Examples:
```bash
python3 main.py --seed=0 --run_num=0 --yaml_file='cheetah_forward'
python3 main.py --seed=0 --run_num=1 --yaml_file='swimmer_forward'
python3 main.py --seed=0 --run_num=2 --yaml_file='ant_forward'
python3 main.py --seed=0 --run_num=3 --yaml_file='hopper_forward'
```
```bash
python3 main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='straight' --visualize_MPC_rollout
python3 main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='backward' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
python3 main.py --seed=0 --run_num=4 --yaml_file='cheetah_trajfollow' --desired_traj_type='forwardbackward' --visualize_MPC_rollout --use_existing_training_data --use_existing_dynamics_model
```
(Additional examples from original README would follow here)
---------------------------------------------------------------

### How to run MBMF

Need to specify:

&nbsp;&nbsp;&nbsp;&nbsp;--**save_trpo_run_num number** Number used as part of directory name for saving mbmf TRPO run (you can use 1,2,3,etc to differentiate your different seeds)
&nbsp;&nbsp;&nbsp;&nbsp;--**run_num** Specify what directory to get relevant MB data from & to save new MBMF files in
&nbsp;&nbsp;&nbsp;&nbsp;--**which_agent** Specify which agent (1 ant, 2 swimmer, 4 cheetah, 6 hopper)
&nbsp;&nbsp;&nbsp;&nbsp;--**std_on_mlp_policy** Initial std you want to set on your pre-initialization policy for TRPO to use
# &nbsp;&nbsp;&nbsp;&nbsp;--**num_workers_trpo** How many worker threads (cpu) for TRPO to use (Note: Parallelism in Garage is configured differently, typically via the sampler in the algorithm or runner.)
&nbsp;&nbsp;&nbsp;&nbsp;--**might_render** If you might want to visualize anything during the run
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_mlp_policy** To visualize the rollout performed by trained policy (that will serve as pre-initialization for TRPO)
&nbsp;&nbsp;&nbsp;&nbsp;--**visualize_on_policy_rollouts** To set a breakpoint and visualize the on-policy rollouts after each agg iteration of dagger
&nbsp;&nbsp;&nbsp;&nbsp;--**print_minimal** To not print messages regarding progress/notes/etc.
&nbsp;&nbsp;&nbsp;&nbsp;--**use_existing_pretrained_policy** To run only the TRPO part (if you've already done the imitation learning part in the same directory)

*Note that the finished TRPO run saves to ~/rllab/data/local/experiments/*

##### Examples:
```bash
python3 mbmf.py --run_num=1 --which_agent=2 --std_on_mlp_policy=1.0
python3 mbmf.py --run_num=0 --which_agent=4 --std_on_mlp_policy=0.5
python3 mbmf.py --run_num=3 --which_agent=6 --std_on_mlp_policy=1.0
python3 mbmf.py --run_num=2 --which_agent=1 --std_on_mlp_policy=0.5
```
---------------------------------------------------------------

### How to run MF

Run pure TRPO, for comparisons.

Need to specify command line args as desired
&nbsp;&nbsp;&nbsp;&nbsp;--**seed** Set random seed to set for numpy and tensorflow
&nbsp;&nbsp;&nbsp;&nbsp;--**steps_per_rollout** Length of each rollout that TRPO should collect
&nbsp;&nbsp;&nbsp;&nbsp;--**save_trpo_run_num** Number used as part of directory name for saving TRPO run (you can use 1,2,3,etc to differentiate your different seeds)
&nbsp;&nbsp;&nbsp;&nbsp;--**which_agent** Specify which agent (1 ant, 2 swimmer, 4 cheetah, 6 hopper)
# &nbsp;&nbsp;&nbsp;&nbsp;--**num_workers_trpo** How many worker threads (cpu) for TRPO to use (Note: Parallelism in Garage is configured differently.)
&nbsp;&nbsp;&nbsp;&nbsp;--**num_trpo_iters** Total number of TRPO iterations to run before stopping

*Note that the finished TRPO run with Garage saves to `data/local/experiment/EXP_NAME` by default, where `EXP_NAME` is generated by the script (e.g., `agent_2_seed_0_mf_run_1`).*

##### Examples:
```bash
python3 trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4
python3 trpo_run_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=2
# (Additional examples from original README would follow here, ensure --num_workers_trpo is removed from them too)
```
---------------------------------------------------------------

### How to plot

1) MBMF
&nbsp;&nbsp;&nbsp;&nbsp;-Need to specify the commandline arguments as desired (in plot_mbmf.py)
&nbsp;&nbsp;&nbsp;&nbsp;-Examples of running the plotting script:
```bash
cd plotting
python3 plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=1.0 --which_agent=2 --run_nums 1 --seeds 0
python3 plot_mbmf.py --trpo_dir=[trpo_dir] --std_on_mlp_policy=1.0 --which_agent=2 --run_nums 1 2 3 --seeds 0 70 100
```
Note that `[trpo_dir]` above corresponds to where the TRPO runs are saved by Garage. This is typically in a subfolder under `data/local/experiment/` within your project directory (e.g., `../data/local/experiment/` if running from `plotting` directory, looking for an experiment like `agent_2_seed_0_mf_run_1`).

2) Dynamics model training and validation losses per aggregation iteration
IPython notebook: plotting/plot_loss.ipynb
Example plots: docs/sample_plots/...

3) Visualize a forward simulation (an open-loop multi-step prediction of the elements of the state space)
IPython notebook: plotting/plot_forwardsim.ipynb
Example plots: docs/sample_plots/...

4) Visualize the trajectories (on policy rollouts) per aggregation iteration
IPython notebook: plotting/plot_trajfollow.ipynb
Example plots: docs/sample_plots/...
```
