import sys
import os
import torch # main.py uses torch
import numpy # main.py uses numpy

# Clean up previous run directory if it exists
run_dir_to_clean = "run_1"
if os.path.exists(run_dir_to_clean):
    print(f"Attempting to clean up existing directory: {run_dir_to_clean}")
    # Using os.system for rm -rf as shutil.rmtree might not be available or might have issues
    # This is a common workaround in restricted environments.
    # Make sure this command is safe and specific.
    rm_command = f"rm -rf {run_dir_to_clean}"
    exit_code = os.system(rm_command)
    if exit_code == 0:
        print(f"Successfully removed directory: {run_dir_to_clean}")
    else:
        print(f"Warning: Failed to remove directory {run_dir_to_clean} with exit code {exit_code}. It might interfere with the test.")

sys.argv = [
    "main.py",
    "--seed=0",
    "--run_num=1",
    "--yaml_file=swimmer_forward",
    "--print_minimal=False",
    "--num_aggregation_iters=1",
    "--num_rollouts_train=2",
    "--num_rollouts_val=1",
    "--steps_per_rollout_train=50",
    "--steps_per_rollout_val=50",
    "--nEpoch=2",
    "--num_trajectories_for_aggregation=2",
    "--rollouts_forTraining=1",
    "--num_rollouts_save_for_mf=1"
]

print(f"sys.argv set to: {sys.argv}")

try:
    from main import main as main_function
    print("Successfully imported main function from main.py. Executing now...")
    main_function()
    print("main_function from main.py executed.")
except Exception as e:
    import traceback
    print(f"An error occurred during import or execution of main.py: {e}")
    print(traceback.format_exc())
    # Propagate the error to ensure the calling bash script knows something went wrong
    sys.exit(1)
