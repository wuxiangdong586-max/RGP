# Reinforcement Genetic Programming (RGP)

This repository contains the official implementation of the experimental framework for the paper: **"Progressive Subexpression Reuse in Symbolic Regression: Insights from RL-based Search and a Genetic Programming Realization"**.

The framework is a Reinforcement Genetic Programming implementation for symbolic regression. It leverages an exploration-exploitation mechanism to iteratively extract and inject high-frequency sub-expressions (mathematical building blocks) back into the evolutionary process, dramatically improving the recovery rate and convergence speed for symbolic regression tasks.

---

## 📁 Project Structure

```
RGP/
├── core/                           # Core algorithmic implementations
│   ├── srl_sr.py                   # Main RGP algorithm and gene injection logic
│   ├── gp.py                       # Base Genetic Programming setup (DEAP wrapper)
│   ├── exploration.py              # Frequency-driven sub-expression exploration
│   ├── exploitation.py             # High-frequency sub-expression mining and extraction
│   ├── function.py                 # Math operators, sequence parsing, and syntax validation
│   └── data_generator.py           # Dataset generator using SymPy
├── configs/                        # Configuration files
│   ├── config.json                 # Config for full experimental suite
│   └── config_easy.json            # Config for single-task quick start
├── dataset/                        # Benchmark datasets
│   └── dataset.csv                 # Ground truth symbolic regression benchmark tasks
├── easy_start.py                   # Minimal entry point for testing a single task
├── main_experiments.py             # Full automated experimental framework (supports multiprocessing)
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Environment Setup

We recommend using Python 3.9+. To install all the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `deap` (Distributed Evolutionary Algorithms in Python)
- `numpy`, `scipy`, `pandas`
- `sympy` (For mathematical simplification and data generation)
- `tqdm` (For progress bars)

---

## 🚀 How to Run

### 1. Quick Start (Single Task)

To run a symbolic regression test on a single benchmark task (e.g., `nguyen-1`) from the terminal:

```bash
python easy_start.py nguyen-1
```

If successful, the script will print out the recovered expression in a human-readable mathematical format (e.g., `x^3 + x^2 + x`).

You can specify a custom configuration file or dataset using arguments:
```bash
python easy_start.py nguyen-1 --seed 42 --config configs/config_easy.json
```

### 2. Full Automated Experiments (Parallel)

To run the full suite of experiments across multiple tasks and random seeds (as reported in the paper), you can use the multiprocessing-enabled main script.

First, check `configs/config.json` to ensure the target `task_list`, `seed_num`, and `max_workers` (number of CPU cores to use) are set to your preference.

Then, start the experiment:

```bash
python main_experiments.py
```

**Output**:
The script will automatically create a timestamped folder inside a `log/` directory. Inside, you will find:
- `recovery.txt`: Success rate and detailed recovered equations per seed.
- `timecost.txt`: Time consumed and the number of generations per seed.
- `non_success.txt`: Failed instances along with the top-k consensus sub-expressions from the last generation for failure analysis.

---

## 📖 Citation

If you find this code useful in your research, please consider citing our paper:
> "Progressive Subexpression Reuse in Symbolic Regression: Insights from RL-based Search and a Genetic Programming Realization"
