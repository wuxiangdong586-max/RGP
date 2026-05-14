# main_experiments.py
"""
Fully Automated Reinforcement Genetic Programming (RGP) Experimental Framework (Supports Parallelism)
- Reads configuration from config.json
- Generates data and runs tasks according to random seeds
- Collects metrics such as recovery rate, time cost, number of rounds, etc.
- Writes results to a log directory named by timestamp
- Supports multi-process parallel execution
"""

import json
import os
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import random
import multiprocessing
from tqdm import tqdm
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify

# Dependencies
from core.data_generator import generate_data_by_name
from core.srl_sr import ReinforcementGeneticProgramming, Config

# =========================
# Utility Functions
# =========================

def progress_print(info):
    r = info["round"] + 1
    print(f"Round {r} | Best MSE = {info['best_fitness']:.6e}")
    print(f"Best Expression: {info['best_expr_str']}")
    if info["topk_common_subexpr"]:
        pretty = [" ".join(tokens) for tokens in info["topk_common_subexpr"]]
        print("Top-k frequent subexpressions in this round:")
        for i, s in enumerate(pretty, 1):
            print(f"  {i}. {s}")
    else:
        print("Top-k frequent subexpressions in this round: None")
    print()

def beautify_expression(expr_str: str) -> str:
    """
    Converts DEAP expression strings (e.g., add(mul(x_1, x_1), x_1)) to SymPy-friendly format.
    """
    try:
        from sympy import symbols, log, sqrt, sin, cos, tan, exp, simplify
        import sympy as sp

        # Extract variable names x_1, x_2, ...
        import re
        var_names = re.findall(r'x_\d+', expr_str)
        vars = {name: symbols(name) for name in set(var_names)}

        # Define function mapping: explicitly define semantics for add, sub, mul, div, pow
        local_dict = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,      # Explicitly a - b, not -(...)
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'pow': lambda a, b: a ** b,
            'log': log,
            'sqrt': sqrt,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'exp': exp,
            **vars
        }

        # Parse expression
        expr = sp.sympify(expr_str, locals=local_dict)
        # Simplify
        expr_simplified = simplify(expr)
        # Convert to string
        result = str(expr_simplified)
        # Optional: display with ^ instead of **
        result = result.replace('**', '^')
        return result

    except Exception as e:
        # If an error occurs, return the original expression
        print(f"Error beautifying expression: {e}")
        return expr_str

def create_timestamp_dir(base_log_dir: str) -> str:
    """Create a directory named with the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_result_files(
    task_name: str,
    log_task_dir: str,
    results: List[Dict]
):
    """Save three result files for a single task"""
    recovery_results = [r["success"] for r in results]
    time_costs = [r["time_cost"] for r in results]
    recovery_rounds = [r["rounds"] for r in results]
    non_success_consensus = [(r["last_consensus"], r["seed"]) for r in results if not r["success"]]

    # 1. Recovery rate file
    with open(os.path.join(log_task_dir, "recovery.txt"), "w") as f:
        total = len(recovery_results)
        success = sum(recovery_results)
        recovery_rate = success / total if total > 0 else 0
        f.write(f"Task: {task_name}\n")
        f.write(f"Total Runs: {total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Recovery Rate: {recovery_rate:.4f}\n")
        f.write("\nDetailed:\n")
        for r in results:
            f.write(f"Seed {r['seed']}: {'Success' if r['success'] else 'Fail'}\n")
            # Beautify expression
            pretty_expr = beautify_expression(r['best_expr_str'])
            f.write(f"  Recovered Expression: {pretty_expr}\n")
            f.write(f"  MSE: {r['best_fitness']:.6e}\n")
            f.write("\n")

    # 2. Time and rounds file
    with open(os.path.join(log_task_dir, "timecost.txt"), "w") as f:
        f.write("seed,time_cost,rounds\n")
        for r in results:
            f.write(f"{r['seed']},{r['time_cost']:.3f},{r['rounds']}\n")

    # 3. Consensus expressions for unsuccessful recoveries
    with open(os.path.join(log_task_dir, "non_success.txt"), "w") as f:
        if not non_success_consensus:
            f.write("All runs succeeded.\n")
        else:
            for consensus, seed in non_success_consensus:
                f.write(f"Seed: {seed}\n")
                f.write("Last Round Top-k Consensus:\n")
                for i, expr in enumerate(consensus, 1):
                    f.write(f"  {i}. {' '.join(expr)}\n")
                f.write("\n")

# =========================
# Single Execution Function (Parallelizable)
# =========================
def run_single_experiment(args) -> Dict:
    """
    Single experiment (can be called by multiple processes)
    Args:
        args: (task_name, config_dict, seed)
    Returns:
        result dict
    """
    # Set the random seed for this process
    random.seed(0)

    task_name, config_dict, seed = args
    np.random.seed(seed)

    # Create Config
    config = Config()
    gp_params = config_dict["gp"]["params"]
    for k, v in gp_params.items():
        config.gp_params[k] = v
    config.num_rounds = config_dict["gp"]["num_rounds"]
    config.initial_random_size = config_dict["gp"]["initial_random_size"]
    config.output_dir = config_dict["gp"]["output_dir"]

    modules = config_dict["modules"]
    config.explore_min_L = modules["exploration"]["min_L"]
    config.explore_max_L = modules["exploration"]["max_L"]
    config.exploit_min_L = modules["exploitation"]["min_L"]
    config.exploit_max_L = modules["exploitation"]["max_L"]
    config.top_k_common = modules["exploitation"]["top_k_common"]
    config.hof_top_k_for_mining = modules["exploitation"]["hof_top_k_for_mining"]
    config.quality_top_percent = modules["exploitation"]["quality_top_percent"]
    config.enable_trig_nesting = modules["validation"]["enable_trig_nesting"]
    config.enable_exp_log_adjacent = modules["validation"]["enable_exp_log_adjacent"]
    config.enable_exp_log_trig = modules["validation"]["enable_exp_log_trig"]
    config.fuse_n = modules["gene_injection"]["fuse_n"]
    config.fuse_m = modules["gene_injection"]["fuse_m"]
    config.inject_prob = modules["gene_injection"]["inject_prob"]
    config.global_hof_size = modules["global_hof_size"]
    config.max_explore_batch = modules["max_explore_batch"]

    try:
        # Generate data
        X, y, _ = generate_data_by_name(task_name, "dataset/dataset.csv", seed=seed)
        # Create model
        rgp = ReinforcementGeneticProgramming(config)
        # Record time
        start_time = time.time()
        # Run (no callback passed to avoid chaotic multi-process printing)
        result = rgp.run(X, y, on_round_end=None)
        end_time = time.time()
        time_cost = end_time - start_time
        # Determine success
        threshold = config.gp_params['threshold']
        success = result['best_fitness'] < threshold
        round_num = result['round'] if success else -1
        # Get the last round's consensus (used for failure analysis)
        last_consensus = []
        if not success and rgp.history:
            last_consensus = rgp.history[-1]['topk_common_subexpr']

        return {
            "seed": seed,
            "success": success,
            "time_cost": time_cost,
            "rounds": round_num,
            "last_consensus": last_consensus,
            "best_expr_str": result['best_expr_str'],
            "best_fitness": result['best_fitness'],
            "error": None
        }

    except Exception as e:
        return {
            "seed": seed,
            "success": False,
            "time_cost": 0.0,
            "rounds": -1,
            "last_consensus": [],
            "best_expr_str": "",
            "best_fitness": float('inf'),
            "error": str(e)
        }

# =========================
# Main Function
# =========================
def run_symbolic_regression_experiment(config_file: str = "configs/config.json"):
    """Main experiment function (supports parallelism)"""
    # 1. Read configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Output CPU information
    total_cpus = multiprocessing.cpu_count()
    used_cpus = config_dict.get("parallel", {}).get("max_workers", total_cpus)
    print(f"🖥️  CPU Info: Using {used_cpus} / {total_cpus} cores")

    # 2. Create log directory
    log_base_dir = "log"
    log_root_dir = create_timestamp_dir(log_base_dir)
    print(f"📝 Experiment logs will be saved to: {log_root_dir}")

    def clean_dir_name(name: str) -> str:
        """Remove invalid characters for Windows filenames"""
        invalid_chars = '/\\:*?"<>|'
        for c in invalid_chars:
            name = name.replace(c, '_')  # Replace invalid characters with underscore
        return name
    # 3. Iterate through each task
    for task_name in config_dict["task"]["task_list"]:
        print(f"\n🚀 Starting task: {task_name}")
        cleaned_task_name = clean_dir_name(task_name)
        if cleaned_task_name != task_name:
            print(f"    Note: Task name contains invalid characters, automatically corrected to: {cleaned_task_name}")
        task_log_dir = os.path.join(log_root_dir, cleaned_task_name)
        os.makedirs(task_log_dir, exist_ok=True)

        # Task parameters
        seed_num = config_dict["task"]["seed_num"]
        seed_shift = config_dict["task"]["seed_shift"]

        # Prepare parameter list
        args_list = [
            (task_name, config_dict, seed_shift + i)
            for i in range(seed_num)
        ]

        # Initialize results
        results = []

        # Get parallelism parameters
        max_workers = config_dict.get("parallel", {}).get("max_workers", None)

        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(run_single_experiment, args): args for args in args_list}
            # Wrap as_completed with tqdm
            for future in tqdm(as_completed(future_to_args), total=len(future_to_args), desc=f"  📊 Running {task_name}", unit="seed"):
                result = future.result()
                results.append(result)
                if result["error"]:
                    print(f"    ❌ Seed {result['seed']} failed: {result['error']}")

        # Save results
        save_result_files(
            task_name=task_name,
            log_task_dir=task_log_dir,
            results=results
        )

        print(f"  ✅ Task {task_name} completed, results saved")

    print(f"\n🎉 All tasks completed! Total log directory: {log_root_dir}")

# =========================
# Example Run
# =========================
if __name__ == "__main__":
    # Set main process random seed
    random.seed(0)
    run_symbolic_regression_experiment("configs/config.json")
