import os
import sys

# --- Determine script and project paths ---
# generate_launchers.py is expected to be in ATCI-P1/scripts/
# configs.py is expected to be in ATCI-P1/src/configs.py
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Should be ATCI-P1/scripts/
PROJECT_ROOT_DIR = os.path.dirname(current_script_dir)          # Should be ATCI-P1/
src_module_parent_dir = PROJECT_ROOT_DIR                         # ATCI-P1/, so we can do "from src import ..."

# --- Attempt to import CONFIGS ---
sys.path.insert(0, src_module_parent_dir)
try:
    from src.configs import CONFIGS # Expects ATCI-P1/src/configs.py
    print(f"Successfully imported CONFIGS from 'src.configs' (expected path: {os.path.join(PROJECT_ROOT_DIR, 'src', 'configs.py')})")
except ImportError as e:
    print(f"Error: Could not import 'src.configs'.")
    print(f"  - This script ('generate_launchers.py') is expected to be in a 'scripts' subdirectory of your project root.")
    print(f"  - 'configs.py' is expected to be in a 'src' subdirectory of your project root (e.g., YOUR_PROJECT_ROOT/src/configs.py).")
    print(f"  Detected script directory: {current_script_dir}")
    print(f"  Deduced project root: {PROJECT_ROOT_DIR}")
    print(f"  Attempted to add '{src_module_parent_dir}' to sys.path to find the 'src' module.")
    print(f"  Import error details: {e}")
    sys.exit(1)
except Exception as e_other:
    print(f"An unexpected error occurred during CONFIGS import: {e_other}")
    sys.exit(1)


# --- SLURM Template ---
# Adapted for ATCI-P1 project structure.
# Note: qos is now acc_training
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=nct328
#SBATCH --qos=acc_training
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nct/nct01026/ATCI-P1/
#SBATCH --output={output_log_path}
#SBATCH --error={error_log_path}

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
# Ensure your Python environment (e.g., virtual environment) with PyTorch, Pydantic, etc.
# is activated here if not part of the loaded Python module.
# Example: source /path/to/your/venv/bin/activate

cd ~/ATCI-P1/  # Redundant if --chdir is effective, but harmless
python src/bsc_main.py -c {config_key}
"""

# --- Path Definitions ---
# Linux paths (used inside the .sh script)
BASE_PROJECT_DIR_LINUX = "/home/nct/nct01026/ATCI-P1" # Updated project name
BASE_LOG_DIR_LINUX = f"{BASE_PROJECT_DIR_LINUX}/out_logs"

# Local paths (used by this Python script to save .sh files)
# current_script_dir and PROJECT_ROOT_DIR are defined above.
SCRIPT_OUTPUT_DIR_NAME = "generated_slurm_sh_scripts"
# .sh files will be saved in e.g., ATCI-P1/scripts/generated_slurm_sh_scripts/
OUTPUT_PATH_FOR_SH_FILES_LOCAL = os.path.join(current_script_dir, SCRIPT_OUTPUT_DIR_NAME)


def sanitize_for_path_and_job_name(name_str):
    name_str = name_str.replace("_", "-")
    sanitized = ''.join(c if c.isalnum() or c == '-' else '' for c in name_str)
    return sanitized[:30]

def main():
    if not os.path.exists(OUTPUT_PATH_FOR_SH_FILES_LOCAL):
        os.makedirs(OUTPUT_PATH_FOR_SH_FILES_LOCAL)
        print(f"Created directory for .sh files: {OUTPUT_PATH_FOR_SH_FILES_LOCAL}")

    config_keys = sorted(list(CONFIGS.keys()))

    if not config_keys:
        print("No configurations found in CONFIGS dictionary from configs.py. Exiting.")
        return

    all_sbatch_commands = []

    for config_key_original in config_keys:
        print(f"Generating SLURM .sh script for config: {config_key_original}")

        log_and_job_name_part = sanitize_for_path_and_job_name(config_key_original)
        sbatch_job_name = f"dl-{log_and_job_name_part}"

        # --- Construct Linux paths for the SLURM script ---
        log_dir_for_this_config_linux = f"{BASE_LOG_DIR_LINUX}/{log_and_job_name_part}"
        output_log_linux = f"{log_dir_for_this_config_linux}/job_output.log"
        error_log_linux = f"{log_dir_for_this_config_linux}/job_error.log"

        # --- Attempt to create the log directory structure locally (relative to PROJECT_ROOT_DIR) ---
        local_log_dir_to_create = os.path.join(
            PROJECT_ROOT_DIR, # Uses the globally defined PROJECT_ROOT_DIR
            "out_logs",
            log_and_job_name_part
        )
        if not os.path.exists(local_log_dir_to_create):
            try:
                os.makedirs(local_log_dir_to_create)
                # print(f"  Successfully created local log directory: {local_log_dir_to_create}")
            except OSError as e:
                print(f"  Warning: Could not create local log directory {local_log_dir_to_create}: {e}")
                print(f"  Ensure the log directory structure exists or can be created on the HPC: {log_dir_for_this_config_linux}")
        # else:
        #     print(f"  Local log directory already exists: {local_log_dir_to_create}")


        script_content = SLURM_TEMPLATE.format(
            job_name=sbatch_job_name,
            output_log_path=output_log_linux,
            error_log_path=error_log_linux,
            config_key=config_key_original
        )

        sh_script_filename_base = f"run_{log_and_job_name_part}.sh"
        sh_script_full_path_local = os.path.join(OUTPUT_PATH_FOR_SH_FILES_LOCAL, sh_script_filename_base)

        with open(sh_script_full_path_local, "w", newline='\n') as f:
            f.write(script_content)
        os.chmod(sh_script_full_path_local, 0o755)

        # --- Construct sbatch command ---
        # Path to the .sh script relative to the project root (BASE_PROJECT_DIR_LINUX or where sbatch is run from)
        scripts_subdir_name = os.path.basename(current_script_dir) # e.g., "scripts"
        sbatch_script_path_from_project_root = f"{scripts_subdir_name}/{SCRIPT_OUTPUT_DIR_NAME}/{sh_script_filename_base}"
        
        sbatch_command = f"sbatch -A nct328 -q acc_training {sbatch_script_path_from_project_root}"
        all_sbatch_commands.append(sbatch_command)

    print(f"\nSuccessfully generated {len(config_keys)} SLURM .sh scripts in: {OUTPUT_PATH_FOR_SH_FILES_LOCAL}")
    print(f"Log files will be written to subdirectories under: {BASE_LOG_DIR_LINUX} on the HPC.")
    print("  Please ensure these log directories exist or can be created by your user on the HPC before jobs run,")
    print("  especially if the local creation attempt (if any) showed warnings.")


    print(f"\n--- List of sbatch commands to run (assuming execution from project root: {PROJECT_ROOT_DIR} on HPC) ---")
    for cmd in all_sbatch_commands:
        print(cmd)

    commands_file_path = os.path.join(current_script_dir, "all_sbatch_submission_commands.txt")
    with open(commands_file_path, "w", newline='\n') as f:
        for cmd in all_sbatch_commands:
            f.write(cmd + "\n")
    print(f"\nList of sbatch commands also saved to: {commands_file_path}")

if __name__ == "__main__":
    main()