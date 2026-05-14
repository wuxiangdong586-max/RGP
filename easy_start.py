# easy_start.py
"""
Minimal entry point for running one symbolic-regression task.

Usage:
    python easy_start.py nguyen-1

On success, stdout contains only the recovered expression.
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config_easy.json"
DEFAULT_DATASET = ROOT / "dataset" / "dataset.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one symbolic-regression task and print the recovered expression."
    )
    parser.add_argument("task", help='Task name, for example "Nguyen-1" or "nguyen-1".')
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config file path.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset CSV path.")
    return parser.parse_args()


def canonical_task_name(task_name: str, dataset_path: Path) -> str:
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name.lower() == task_name.lower():
                return name
    raise ValueError(f"Task not found in dataset: {task_name}")


def apply_config_file(config, config_path: Path) -> None:
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    gp_config = config_data.get("gp", {})
    config.gp_params.update(gp_config.get("params", {}))
    config.num_rounds = gp_config.get("num_rounds", config.num_rounds)
    config.initial_random_size = gp_config.get("initial_random_size", config.initial_random_size)
    config.output_dir = gp_config.get("output_dir", config.output_dir)

    modules = config_data.get("modules", {})
    exploration = modules.get("exploration", {})
    exploitation = modules.get("exploitation", {})
    validation = modules.get("validation", {})
    gene_injection = modules.get("gene_injection", {})

    config.explore_min_L = exploration.get("min_L", config.explore_min_L)
    config.explore_max_L = exploration.get("max_L", config.explore_max_L)
    config.exploit_min_L = exploitation.get("min_L", config.exploit_min_L)
    config.exploit_max_L = exploitation.get("max_L", config.exploit_max_L)
    config.top_k_common = exploitation.get("top_k_common", config.top_k_common)
    config.hof_top_k_for_mining = exploitation.get(
        "hof_top_k_for_mining", config.hof_top_k_for_mining
    )
    config.quality_top_percent = exploitation.get(
        "quality_top_percent", config.quality_top_percent
    )
    config.enable_trig_nesting = validation.get(
        "enable_trig_nesting", config.enable_trig_nesting
    )
    config.enable_exp_log_adjacent = validation.get(
        "enable_exp_log_adjacent", config.enable_exp_log_adjacent
    )
    config.enable_exp_log_trig = validation.get(
        "enable_exp_log_trig", config.enable_exp_log_trig
    )
    config.fuse_n = gene_injection.get("fuse_n", config.fuse_n)
    config.fuse_m = gene_injection.get("fuse_m", config.fuse_m)
    config.inject_prob = gene_injection.get("inject_prob", config.inject_prob)
    config.global_hof_size = modules.get("global_hof_size", config.global_hof_size)
    config.max_explore_batch = modules.get("max_explore_batch", config.max_explore_batch)


def format_recovered_expression(expr_str: str) -> str:
    try:
        import sympy as sp

        var_names = sorted(
            set(re.findall(r"x_\d+", expr_str)),
            key=lambda name: int(name.split("_")[1]),
        )
        pretty_names = ["x", "y", "z", "u", "v", "w"]
        variables = {
            name: sp.Symbol(pretty_names[i] if i < len(pretty_names) else f"x{i + 1}")
            for i, name in enumerate(var_names)
        }
        local_dict = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "log": sp.log,
            **variables,
        }
        expr = sp.sympify(expr_str, locals=local_dict)
        formatted = sp.expand(sp.simplify(expr))
        return str(formatted).replace("**", "^")
    except Exception:
        return expr_str


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    config_path = Path(args.config)

    try:
        import numpy as np

        from core.data_generator import generate_data_by_name
        from core.srl_sr import Config, ReinforcementGeneticProgramming
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. Install dependencies with: "
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        task_name = canonical_task_name(args.task, dataset_path)
        random.seed(args.seed)
        np.random.seed(args.seed)

        X, y, _ = generate_data_by_name(task_name, str(dataset_path), seed=args.seed)

        config = Config()
        apply_config_file(config, config_path)
        config.input_dim = X.shape[1]
        config.n_samples = X.shape[0]

        model = ReinforcementGeneticProgramming(config)
        model.save_hall_of_fame = lambda *_args, **_kwargs: None

        result = model.run(X, y, on_round_end=None)
        print(format_recovered_expression(result["best_expr_str"]))
        return 0
    except Exception as exc:
        print(f"easy_start failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
