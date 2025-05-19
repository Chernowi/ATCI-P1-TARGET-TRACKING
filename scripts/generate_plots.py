import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import glob
import seaborn as sns
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ALL_CONFIG_NAMES = [
    "default", "default_poor_signal", "default_good_signal",
    "sac_rnn", "sac_per", "ppo_mlp", "ppo_rnn",
    "sac_mlp_actor_lr_low", "sac_mlp_actor_lr_high",
    "sac_mlp_critic_lr_low", "sac_mlp_critic_lr_high",
    "sac_mlp_gamma_low", "sac_mlp_gamma_high",
    "sac_mlp_tau_low", "sac_mlp_tau_high",
    "sac_mlp_hidden_dims_small", "sac_mlp_hidden_dims_large",
    "sac_mlp_alpha_low", "sac_mlp_alpha_high",
    "ppo_mlp_actor_lr_low", "ppo_mlp_actor_lr_high",
    "ppo_mlp_critic_lr_low", "ppo_mlp_critic_lr_high",
    "ppo_mlp_gae_lambda_low", "ppo_mlp_gae_lambda_high",
    "ppo_mlp_policy_clip_low", "ppo_mlp_policy_clip_high",
    "ppo_mlp_entropy_coef_low", "ppo_mlp_entropy_coef_high",
    "ppo_mlp_hidden_dim_small", "ppo_mlp_hidden_dim_large",
    "ppo_mlp_n_epochs_low", "ppo_mlp_n_epochs_high",
]

ALGO_SAC = "sac"
ALGO_PPO = "ppo"

REWARD_AVG_100_TAG = "Reward/Average_100_Raw"
REWARD_EPISODE_TAG = "Reward/Episode_Raw"
PERFORMANCE_METRIC_TAG = "Error/Distance_EndEp" # Error metric

palette = sns.color_palette("deep", 10)
CONSISTENT_LINESTYLE = '-'
STD_WINDOW_SIZE = 20

GENERAL_STEPS_PLOT_X_MAX_MILLIONS = None
GENERAL_TIME_PLOT_X_MAX_HOURS = None
ERROR_STEPS_PLOT_X_MAX_MILLIONS = 20
ERROR_TIME_PLOT_X_MAX_HOURS = None


# --- Helper Functions ---
def infer_algo_from_config_name(config_name):
    if "ppo" in config_name: return ALGO_PPO
    if "sac" in config_name or config_name == "default" or config_name.startswith("default_"): return ALGO_SAC
    return None

def find_latest_experiment_folder(base_dir, algo_name, config_key_name):
    pattern = os.path.join(base_dir, f"{algo_name}_{config_key_name}_exp_*")
    folders = glob.glob(pattern)
    if not folders: return None
    valid_folders = [f for f in folders if os.path.basename(f).split('_')[-2] == "exp" and os.path.basename(f).split('_')[-1].isdigit() and os.path.basename(f).startswith(f"{algo_name}_{config_key_name}_exp_")]
    if not valid_folders: return None
    try:
        valid_folders.sort(key=lambda f: int(os.path.basename(f).split('_')[-1]))
        return valid_folders[-1]
    except ValueError:
        return valid_folders[-1] if valid_folders else None

def extract_scalar_data(event_file_path, scalar_tags_list):
    data = {tag: {'wall_times': np.array([]), 'steps': np.array([]), 'values': np.array([])} for tag in scalar_tags_list}
    try:
        ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        for tag in scalar_tags_list:
            if tag in available_tags:
                events = ea.Scalars(tag)
                data[tag]['wall_times'] = np.array([event.wall_time for event in events])
                data[tag]['steps'] = np.array([event.step for event in events])
                data[tag]['values'] = np.array([event.value for event in events])
    except Exception as e: print(f"Error processing event file {event_file_path}: {e}")
    return data

def get_plot_data(exp_data, y_tag, x_axis_type="env_steps", std_source_y_tag=None, average_window_main_line=None):
    y_values_raw = exp_data.get(y_tag, {}).get('values', np.array([]))
    x_orig_steps_for_y_tag = exp_data.get(y_tag, {}).get('steps', np.array([]))
    wall_times_raw_for_y_tag = exp_data.get(y_tag, {}).get('wall_times', np.array([]))

    # Apply absolute value if it's the error metric for the main line
    y_values_processed = np.abs(y_values_raw) if y_tag == PERFORMANCE_METRIC_TAG else y_values_raw

    std_source_y_tag_resolved = std_source_y_tag if std_source_y_tag else y_tag
    y_values_for_std_calc_raw = exp_data.get(std_source_y_tag_resolved, {}).get('values', np.array([]))
    
    # Apply absolute value if it's the error metric for the std source
    y_values_for_std_calc_processed = np.abs(y_values_for_std_calc_raw) if std_source_y_tag_resolved == PERFORMANCE_METRIC_TAG else y_values_for_std_calc_raw
    
    x_orig_steps_for_std_calc = exp_data.get(std_source_y_tag_resolved, {}).get('steps', np.array([]))
    wall_times_raw_for_std_calc = exp_data.get(std_source_y_tag_resolved, {}).get('wall_times', np.array([]))
    
    x_plot_final, y_plot_main, std_plot_final = np.array([]), np.array([]), None

    if x_axis_type == "env_steps":
        x_plot_final, y_plot_main = x_orig_steps_for_y_tag, y_values_processed
        x_for_std_dev_source = x_orig_steps_for_std_calc
    elif x_axis_type == "time":
        if wall_times_raw_for_y_tag.size > 0:
            x_plot_final, y_plot_main = (wall_times_raw_for_y_tag - wall_times_raw_for_y_tag[0]) / 3600.0, y_values_processed
        else: return np.array([]), np.array([]), None
        x_for_std_dev_source = (wall_times_raw_for_std_calc - wall_times_raw_for_std_calc[0]) / 3600.0 if wall_times_raw_for_std_calc.size > 0 else np.array([])
    else: return np.array([]), np.array([]), None

    if not (x_plot_final.size > 0 and y_plot_main.size == x_plot_final.size): return np.array([]), np.array([]), None
    sort_idx_main = np.argsort(x_plot_final)
    x_plot_final, y_plot_main = x_plot_final[sort_idx_main], y_plot_main[sort_idx_main]

    if average_window_main_line and isinstance(average_window_main_line, int) and average_window_main_line > 0 and y_plot_main.size >= 1:
        y_plot_main = pd.Series(y_plot_main).rolling(window=average_window_main_line, min_periods=1).mean().to_numpy()

    if y_values_for_std_calc_processed.size >= STD_WINDOW_SIZE and x_for_std_dev_source.size == y_values_for_std_calc_processed.size:
        sort_idx_std_src = np.argsort(x_for_std_dev_source)
        x_std_src_sorted, y_std_src_sorted = x_for_std_dev_source[sort_idx_std_src], y_values_for_std_calc_processed[sort_idx_std_src]
        rolling_std = pd.Series(y_std_src_sorted).rolling(window=STD_WINDOW_SIZE, min_periods=1).std().to_numpy()
        if x_plot_final.size > 0 and x_std_src_sorted.size > 0:
            std_plot_final = np.interp(x_plot_final, x_std_src_sorted, rolling_std, left=np.nan, right=np.nan)
            std_plot_final = pd.Series(std_plot_final).fillna(method='bfill').fillna(method='ffill').to_numpy()
    return x_plot_final, y_plot_main, std_plot_final

def plot_data_with_std_shade(plot_title, xlabel, ylabel, output_filename, datasets,
                             x_is_million_steps=False, x_max_limit_resolved=None,
                             y_max_limit=None, y_min_limit=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    legend_handles_labels = {}
    max_x_val_observed, min_x_val_observed, all_y_data_for_ylim = 0, float('inf'), []

    for x_data, y_data, std_data, label, color, linestyle in datasets:
        if not (x_data.size > 0 and y_data.size > 0 and len(x_data) == len(y_data)): continue
        x_plot = x_data / 1e6 if x_is_million_steps else x_data
        if x_plot.size > 0:
            max_x_val_observed, min_x_val_observed = max(max_x_val_observed, np.max(x_plot)), min(min_x_val_observed, np.min(x_plot))
            all_y_data_for_ylim.extend(y_data)
            if std_data is not None and std_data.size == y_data.size: all_y_data_for_ylim.extend(y_data - std_data); all_y_data_for_ylim.extend(y_data + std_data)
        
        x_plot_filtered, y_data_filtered, std_data_filtered = x_plot, y_data, std_data
        if x_max_limit_resolved is not None and x_plot.size > 0:
            mask = x_plot <= x_max_limit_resolved
            x_plot_filtered, y_data_filtered = x_plot[mask], y_data[mask]
            if std_data is not None and std_data.size > 0 : std_data_filtered = std_data[mask[:len(std_data)]]
            else: std_data_filtered = None
        if not (x_plot_filtered.size > 0 and y_data_filtered.size > 0): continue

        line, = plt.plot(x_plot_filtered, y_data_filtered, color=color, linestyle=linestyle, alpha=0.95, linewidth=2.0, label=label)
        if label not in legend_handles_labels: legend_handles_labels[label] = line 
        elif legend_handles_labels[label].get_color() != color : 
             legend_handles_labels[label+f"_{color}"] = line # Should not happen with current simple SAC/PPO labels for combined plot


        if std_data_filtered is not None and std_data_filtered.size == y_data_filtered.size:
            plt.fill_between(x_plot_filtered, np.nan_to_num(y_data_filtered - std_data_filtered), np.nan_to_num(y_data_filtered + std_data_filtered), color=color, alpha=0.20)

    plt.title(plot_title, fontsize=16, fontweight='bold'); plt.xlabel(xlabel, fontsize=14); plt.ylabel(ylabel, fontsize=14)
    if legend_handles_labels:
        legend = plt.legend(legend_handles_labels.values(), legend_handles_labels.keys(), fontsize=11, loc='best')
        if legend: legend.get_frame().set_alpha(0.95)
    plt.grid(True, which="both", ls="--", alpha=0.5); plt.tick_params(axis='both', which='major', labelsize=12)
    left_xlim_val = 0
    if min_x_val_observed != float('inf') and min_x_val_observed < 0 : left_xlim_val = min_x_val_observed
    current_xlim_right = x_max_limit_resolved if x_max_limit_resolved is not None else (max_x_val_observed if max_x_val_observed > 0 else 1) 
    if current_xlim_right <= left_xlim_val : current_xlim_right = left_xlim_val + (1e-6 if x_is_million_steps else 1)
    plt.xlim(left=left_xlim_val, right=current_xlim_right)
    if y_min_limit is not None and y_max_limit is not None: plt.ylim(y_min_limit, y_max_limit)
    elif all_y_data_for_ylim:
        y_min_plot_data, y_max_plot_data = np.nanmin(all_y_data_for_ylim), np.nanmax(all_y_data_for_ylim)
        if not (np.isnan(y_min_plot_data) or np.isnan(y_max_plot_data)) and y_max_plot_data > y_min_plot_data :
            y_padding = (y_max_plot_data - y_min_plot_data) * 0.05
            y_padding = max(y_padding, 0.01 if y_max_plot_data - y_min_plot_data < 0.2 else y_padding) 
            plt.ylim(y_min_plot_data - y_padding, y_max_plot_data + y_padding)
        elif not (np.isnan(y_min_plot_data) or np.isnan(y_max_plot_data)): plt.ylim(y_min_plot_data - 0.1, y_max_plot_data + 0.1)
    plt.tight_layout(); save_path = os.path.join(FIGURES_DIR, output_filename); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Plot saved: {save_path}")


def load_all_experiment_data(config_keys_list):
    print("--- Loading All Experiment Data ---")
    all_data = {}
    for config_key in config_keys_list:
        algo = infer_algo_from_config_name(config_key)
        if not algo: all_data[config_key] = None; continue
        folder = find_latest_experiment_folder(EXPERIMENTS_DIR, algo, config_key)
        if folder:
            event_files = glob.glob(os.path.join(folder, "tensorboard", "events.out.tfevents.*"))
            if event_files:
                tags_to_extract = [REWARD_AVG_100_TAG, REWARD_EPISODE_TAG, PERFORMANCE_METRIC_TAG]
                data = extract_scalar_data(event_files[0], tags_to_extract)
                data["_algo"] = algo; data["_config_key"] = config_key
                all_data[config_key] = data
            else: all_data[config_key] = None
        else: all_data[config_key] = None
    print("--- Data Loading Complete ---")
    return all_data

# --- Plotting Functions ---
def plot_general_comparison(all_exp_data, y_tag_to_plot, y_label, file_suffix, std_source_tag,
                            y_min=None, y_max=None, avg_window_main=None, 
                            x_max_steps_limit=None, x_max_time_limit=None):
    print(f"\n--- Plotting General Algorithm Comparison ({y_label}) ---")
    configs_for_plot = ["default", "ppo_mlp", "sac_rnn", "ppo_rnn"]
    datasets_steps, datasets_time = [], []
    color_map = {"default": palette[1], "ppo_mlp": palette[0], "sac_rnn": palette[3], "ppo_rnn": palette[2]}
    label_map = {"default": "SAC MLP", "ppo_mlp": "PPO MLP", "sac_rnn": "SAC RNN", "ppo_rnn": "PPO RNN"}

    for i, config_key in enumerate(configs_for_plot):
        exp_data = all_exp_data.get(config_key)
        if not exp_data: continue
        label, color = label_map.get(config_key, config_key), color_map.get(config_key, palette[i % len(palette)])
        x_steps, y_steps, std_steps = get_plot_data(exp_data, y_tag_to_plot, "env_steps", std_source_tag, avg_window_main)
        if x_steps.size > 0: datasets_steps.append((x_steps, y_steps, std_steps, label, color, CONSISTENT_LINESTYLE))
        x_time, y_time, std_time = get_plot_data(exp_data, y_tag_to_plot, "time", std_source_tag, avg_window_main)
        if x_time.size > 0: datasets_time.append((x_time, y_time, std_time, label, color, CONSISTENT_LINESTYLE))

    if datasets_steps:
        plot_data_with_std_shade(f"Algorithm Comparison: {y_label} vs Environment Steps", "Environment Steps (Millions)", y_label,
                                 f"general_compare_{file_suffix}_vs_steps.png", datasets_steps, x_is_million_steps=True, x_max_limit_resolved=x_max_steps_limit, y_max_limit=y_max, y_min_limit=y_min)
    if datasets_time:
        plot_data_with_std_shade(f"Algorithm Comparison: {y_label} vs Running Time", "Running Time (hours)", y_label,
                                 f"general_compare_{file_suffix}_vs_time.png", datasets_time, x_is_million_steps=False, x_max_limit_resolved=x_max_time_limit, y_max_limit=y_max, y_min_limit=y_min)

def plot_hyperparam_comparison(all_exp_data, base_config_key, param_variations_with_values,
                               param_name_short, y_tag_to_plot, y_label, file_suffix_base,
                               std_source_tag, y_min=None, y_max=None, avg_window_main=None,
                               x_max_steps_limit=None, x_max_time_limit=None):
    print(f"\n--- Plotting {param_name_short} Hyperparam Comparison for {base_config_key} ({y_label}) ---")
    base_exp_data = all_exp_data.get(base_config_key)
    if not base_exp_data: print(f"Base config '{base_config_key}' data not found. Skipping."); return

    datasets_steps, datasets_time = [], []
    current_palette = sns.color_palette("husl", len(param_variations_with_values) + 1)
    base_label, base_color = f"{param_name_short}: Default", current_palette[0]
    x_steps_base, y_steps_base, std_steps_base = get_plot_data(base_exp_data, y_tag_to_plot, "env_steps", std_source_tag, avg_window_main)
    if x_steps_base.size > 0: datasets_steps.append((x_steps_base, y_steps_base, std_steps_base, base_label, base_color, CONSISTENT_LINESTYLE))
    x_time_base, y_time_base, std_time_base = get_plot_data(base_exp_data, y_tag_to_plot, "time", std_source_tag, avg_window_main)
    if x_time_base.size > 0: datasets_time.append((x_time_base, y_time_base, std_time_base, base_label, base_color, CONSISTENT_LINESTYLE))

    for i, (var_key, value_display_str) in enumerate(param_variations_with_values):
        var_exp_data = all_exp_data.get(var_key)
        if not var_exp_data: continue
        var_label, var_color = f"{param_name_short}: {value_display_str}", current_palette[i + 1]
        x_steps_var, y_steps_var, std_steps_var = get_plot_data(var_exp_data, y_tag_to_plot, "env_steps", std_source_tag, avg_window_main)
        if x_steps_var.size > 0: datasets_steps.append((x_steps_var, y_steps_var, std_steps_var, var_label, var_color, CONSISTENT_LINESTYLE))
        x_time_var, y_time_var, std_time_var = get_plot_data(var_exp_data, y_tag_to_plot, "time", std_source_tag, avg_window_main)
        if x_time_var.size > 0: datasets_time.append((x_time_var, y_time_var, std_time_var, var_label, var_color, CONSISTENT_LINESTYLE))

    algo_name, arch_name = base_exp_data['_algo'], 'rnn' if 'rnn' in base_config_key else ('mlp' if base_config_key == "default" or "mlp" in base_config_key else "")
    plot_title_prefix = f"{algo_name.upper()} {arch_name.upper()}: {param_name_short} Comparison"
    if len(datasets_steps) > 1:
        plot_data_with_std_shade(f"{plot_title_prefix} - {y_label} vs Steps", "Environment Steps (Millions)", y_label,
                                 f"{algo_name}_{arch_name}_{file_suffix_base}_vs_steps.png", datasets_steps, True, x_max_steps_limit, y_max, y_min)
    if len(datasets_time) > 1:
        plot_data_with_std_shade(f"{plot_title_prefix} - {y_label} vs Time", "Running Time (hours)", y_label,
                                 f"{algo_name}_{arch_name}_{file_suffix_base}_vs_time.png", datasets_time, False, x_max_time_limit, y_max, y_min)

def plot_all_algorithms_combined(all_exp_data, y_tag, y_label, file_suffix, 
                                 std_source_tag, y_min=None, y_max=None, 
                                 avg_window_main=None, x_max_steps_limit=None, x_max_time_limit=None):
    print(f"\n--- Plotting Combined Algorithm Performance ({y_label}) ---")
    datasets_steps, datasets_time = [], []
    sac_color, ppo_color = palette[1], palette[0] 
    
    sac_configs_plotted = 0
    ppo_configs_plotted = 0

    for config_key, exp_data in all_exp_data.items():
        if not exp_data: continue
        algo = exp_data['_algo']
        
        color = sac_color if algo == ALGO_SAC else ppo_color
        label = algo.upper() # Simplified label: "SAC" or "PPO"
        
        x_steps, y_steps, std_steps = get_plot_data(exp_data, y_tag, "env_steps", std_source_tag, avg_window_main)
        if x_steps.size > 0: 
            datasets_steps.append((x_steps, y_steps, std_steps, label, color, CONSISTENT_LINESTYLE))
            if algo == ALGO_SAC: sac_configs_plotted +=1
            else: ppo_configs_plotted +=1
        
        x_time, y_time, std_time = get_plot_data(exp_data, y_tag, "time", std_source_tag, avg_window_main)
        if x_time.size > 0: datasets_time.append((x_time, y_time, std_time, label, color, CONSISTENT_LINESTYLE))

    title_suffix = f" ({sac_configs_plotted} SAC, {ppo_configs_plotted} PPO runs)" if (sac_configs_plotted + ppo_configs_plotted > 0) else ""
    if datasets_steps:
        plot_data_with_std_shade(f"All Configurations: {y_label} vs Environment Steps{title_suffix}", "Environment Steps (Millions)", y_label,
                                 f"all_configs_combined_{file_suffix}_vs_steps.png", datasets_steps, x_is_million_steps=True, x_max_limit_resolved=x_max_steps_limit, y_max_limit=y_max, y_min_limit=y_min)
    if datasets_time:
        plot_data_with_std_shade(f"All Configurations: {y_label} vs Running Time{title_suffix}", "Running Time (hours)", y_label,
                                 f"all_configs_combined_{file_suffix}_vs_time.png", datasets_time, x_is_million_steps=False, x_max_limit_resolved=x_max_time_limit, y_max_limit=y_max, y_min_limit=y_min)


# --- Main Script ---
if __name__ == "__main__":
    all_experiment_data = load_all_experiment_data(ALL_CONFIG_NAMES)

    metrics_to_plot = [
        (REWARD_AVG_100_TAG, "Avg Return (100 Ep)", "reward_avg100", REWARD_EPISODE_TAG, None, None, None),
        (PERFORMANCE_METRIC_TAG, "Avg Absolute End-of-Ep Error (100 Ep)", "abs_errordist_avg100_endep", PERFORMANCE_METRIC_TAG, None, None, 100) # y_min/max for error handled below
    ]

    for y_tag, y_axis_label, file_label_suffix, std_source_tag_for_plot, y_min_default, y_max_default, avg_window in metrics_to_plot:
        
        current_x_max_steps = GENERAL_STEPS_PLOT_X_MAX_MILLIONS
        current_x_max_time = GENERAL_TIME_PLOT_X_MAX_HOURS
        current_y_min = y_min_default
        current_y_max = y_max_default
        current_y_label = y_axis_label

        if y_tag == PERFORMANCE_METRIC_TAG: 
            if ERROR_STEPS_PLOT_X_MAX_MILLIONS is not None:
                current_x_max_steps = ERROR_STEPS_PLOT_X_MAX_MILLIONS
            if ERROR_TIME_PLOT_X_MAX_HOURS is not None:
                current_x_max_time = ERROR_TIME_PLOT_X_MAX_HOURS
            current_y_min = 0  # Force y-min to 0 for error
            current_y_max = 50 # Force y-max to 50 for error
            # Label already updated in metrics_to_plot
        
        plot_all_algorithms_combined(all_experiment_data, y_tag, current_y_label, file_label_suffix,
                                     std_source_tag_for_plot, current_y_min, current_y_max, avg_window,
                                     x_max_steps_limit=current_x_max_steps, x_max_time_limit=current_x_max_time)

        plot_general_comparison(all_experiment_data, y_tag, current_y_label, file_label_suffix,
                                std_source_tag_for_plot, current_y_min, current_y_max, avg_window,
                                x_max_steps_limit=current_x_max_steps, x_max_time_limit=current_x_max_time)
        
        sac_mlp_hyperparams = {
            "Actor LR": ([("sac_mlp_actor_lr_low", "1e-5"), ("sac_mlp_actor_lr_high", "1e-4")], "actor_lr"),
            "Critic LR": ([("sac_mlp_critic_lr_low", "1e-5"), ("sac_mlp_critic_lr_high", "1e-4")], "critic_lr"),
            "Gamma": ([("sac_mlp_gamma_low", "0.95"), ("sac_mlp_gamma_high", "0.995")], "gamma"),
            "Tau": ([("sac_mlp_tau_low", "0.001"), ("sac_mlp_tau_high", "0.01")], "tau"),
            "Hidden Dims": ([("sac_mlp_hidden_dims_small", "[32,32]"), ("sac_mlp_hidden_dims_large", "[128,128]")], "hidden_dims"),
            "Alpha": ([("sac_mlp_alpha_low", "0.1"), ("sac_mlp_alpha_high", "0.5")], "alpha"),
            "PER vs Default": ([("sac_per", "PER Enabled")], "per")
        }
        for param_name, (variations, suffix) in sac_mlp_hyperparams.items():
            plot_hyperparam_comparison(all_experiment_data, "default", variations, param_name, y_tag, current_y_label,
                                       f"{suffix}_{file_label_suffix}", std_source_tag_for_plot, current_y_min, current_y_max, avg_window,
                                       x_max_steps_limit=current_x_max_steps, x_max_time_limit=current_x_max_time)
        
        ppo_mlp_hyperparams = {
            "Actor LR": ([("ppo_mlp_actor_lr_low", "1e-6"), ("ppo_mlp_actor_lr_high", "1e-5")], "actor_lr"),
            "Critic LR": ([("ppo_mlp_critic_lr_low", "5e-4"), ("ppo_mlp_critic_lr_high", "5e-3")], "critic_lr"),
            "GAE Lambda": ([("ppo_mlp_gae_lambda_low", "0.90"), ("ppo_mlp_gae_lambda_high", "0.98")], "gae_lambda"),
            "Policy Clip": ([("ppo_mlp_policy_clip_low", "0.02"), ("ppo_mlp_policy_clip_high", "0.1")], "policy_clip"),
            "Entropy Coef": ([("ppo_mlp_entropy_coef_low", "0.005"), ("ppo_mlp_entropy_coef_high", "0.05")], "entropy_coef"),
            "Hidden Dim": ([("ppo_mlp_hidden_dim_small", "128"), ("ppo_mlp_hidden_dim_large", "512")], "hidden_dim"),
            "Num Epochs": ([("ppo_mlp_n_epochs_low", "2"), ("ppo_mlp_n_epochs_high", "5")], "n_epochs"),
        }
        for param_name, (variations, suffix) in ppo_mlp_hyperparams.items():
            plot_hyperparam_comparison(all_experiment_data, "ppo_mlp", variations, param_name, y_tag, current_y_label,
                                       f"{suffix}_{file_label_suffix}", std_source_tag_for_plot, current_y_min, current_y_max, avg_window,
                                       x_max_steps_limit=current_x_max_steps, x_max_time_limit=current_x_max_time)
        
        signal_variations = { # Renamed to avoid conflict if used for other algos
            "Signal Quality (SAC MLP)": ([("default_poor_signal", "Poor"), ("default_good_signal", "Good")], "signal_quality_sac_mlp")
        }
        for param_name, (variations, suffix) in signal_variations.items():
             plot_hyperparam_comparison(all_experiment_data, "default", variations, param_name, y_tag, current_y_label,
                                       f"{suffix}_{file_label_suffix}", std_source_tag_for_plot, current_y_min, current_y_max, avg_window,
                                       x_max_steps_limit=current_x_max_steps, x_max_time_limit=current_x_max_time)

    print("\nPlotting script finished.")