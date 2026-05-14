# srl_sr.py
"""
Main Control Program: Structured Gene Injection Mutation + Safe Subtree Replacement
- Separate configuration for exploration and exploitation sub-expression length ranges
- Per-round result output: implemented via callback function in main() (printed only in main function)
- High-frequency sub-expression extraction: length merging + quality truncation + frequency sorting
"""

import random
import numpy as np
from deap import gp, creator, base
import pickle
import os
from typing import List, Tuple
from collections import Counter

# === Dynamically create DEAP types ===
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Internal module dependencies
from core.exploration import ExplorationManager
from core.exploitation import SubexpressionExtractor
from core.function import (
    register_operator_functions,
    prefix_to_internal_expression,
    expr_to_sequence,
    ExpressionValidator
)
from core.gp import SymbolicRegression, GP_config


# =========================
# Configuration Parameters (Centralized Management)
# =========================
class Config:
    def __init__(self):
        # === GP Parameters ===
        self.gp_params = {
            "pops": 500,
            "times": 30,
            "tournsize": 5,
            "cxpb": 0.5,
            "mutpb": 0.5,
            "max_height": 10,
            "max_size": 32,
            "hof_size": 30,
            "threshold": 1e-6,
        }

        # === Exploration Module Length Range ===
        self.explore_min_L = 3
        self.explore_max_L = 5

        # === Exploitation Module Length Range ===
        self.exploit_min_L = 3
        self.exploit_max_L = 8

        # === Generation/Legality Switches ===
        self.enable_trig_nesting = True
        self.enable_exp_log_adjacent = True
        self.enable_exp_log_trig = True

        # === Training Rounds and Initial Pool ===
        self.num_rounds = 2000
        self.initial_random_size = 500

        # === Data Parameters ===
        self.n_samples = 100
        self.input_dim = 2

        # === Output Directory ===
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # === Injection and Fusion Hyperparameters ===
        self.fuse_n = 3
        self.fuse_m = 2
        self.inject_prob = 0.5

        # === High-frequency Sub-expression Mining Parameters ===
        self.global_hof_size = 50
        self.top_k_common = 5
        self.hof_top_k_for_mining = 20
        self.max_explore_batch = 20

        # === Quality Truncation Ratio (Top p%) ===
        self.quality_top_percent = 0.5  # Take the top 50% with best average fitness

        # === Exploration Frequency Decay Factor ===
        self.explore_decay_factor = 0.99  # Multiply by 0.99 each round

        # === Restart Patience ===
        self.restart_patience = 30

# =========================
# Main Control Class
# =========================
class ReinforcementGeneticProgramming:
    def __init__(self, config: Config):
        self.config = config
        self.operator_functions = register_operator_functions()
        self.operators = {op: info["arity"] for op, info in self.operator_functions.items()}
        self.operands = [f"x_{i+1}" for i in range(config.input_dim)]

        self.exploration_manager = ExplorationManager(
            min_L=config.explore_min_L,
            max_L=config.explore_max_L,
            operators=self.operators,
            operands=self.operands,
            trig_nesting=config.enable_trig_nesting,
            exp_log_adjacent=config.enable_exp_log_adjacent,
            exp_log_trig=config.enable_exp_log_trig
        )
        self.exploration_manager.generate_candidates()

        symbol_arity = self.operators.copy()
        symbol_arity.update({op: 0 for op in self.operands})
        self.subexpr_extractor = SubexpressionExtractor(symbol_arity=symbol_arity)

        self.gp_config = GP_config(self.config.gp_params)
        self.sr = SymbolicRegression(self.gp_config, self.operator_functions)

        self.X = None
        self.y = None
        self.current_round = 0
        self.history = []

        self.global_hof: List[Tuple[float, creator.Individual]] = []
        self.max_global_hof_size = self.config.global_hof_size

        self.seen_global_exprs = set()
        self.validator = ExpressionValidator(self.operator_functions)

        # Exploration mask for the next round
        self.next_mask_set = set()

        # Restart rules
        self.stable_count = 0
        self.restart_patience = self.config.restart_patience
        self.last_topk = None
    # -------- High-frequency Sub-expression Extraction (Exploitation Range)--------
    def extract_common_subexpressions_from_global_hof(self) -> List[List[str]]:
        """Extract high-frequency common sub-expressions (preorder token list) from global HOF - using exploitation length range"""
        if not self.global_hof:
            return []

        sorted_hof = sorted(self.global_hof, key=lambda x: x[0])
        top_individuals = [ind for _, ind in sorted_hof[: self.config.hof_top_k_for_mining]]

        # Collect frequency and fitness of the individual for each sub-expression
        subexpr_stats = {}  # tuple(expr) -> {freq, fitnesses}

        for ind in top_individuals:
            try:
                internal_str = str(ind)
                prefix_seq = expr_to_sequence(internal_str)
                fitness = ind.fitness.values[0]

                subexpr_tuples = self.subexpr_extractor.extract(
                    expressions=[prefix_seq],
                    min_len=self.config.exploit_min_L,
                    max_len=self.config.exploit_max_L
                )

                for t in subexpr_tuples:
                    key = tuple(t)
                    if key not in subexpr_stats:
                        subexpr_stats[key] = {'freq': 0, 'fitnesses': []}
                    subexpr_stats[key]['freq'] += 1
                    subexpr_stats[key]['fitnesses'].append(fitness)
            except Exception:
                continue

        if not subexpr_stats:
            return []

        # Calculate avg_fitness for each sub-expression
        items = []
        for key, stat in subexpr_stats.items():
            avg_fitness = np.mean(stat['fitnesses'])
            freq = stat['freq']
            items.append((key, avg_fitness, freq))

        if len(items) <= 1:
            return [list(items[0][0])] if items else []

        # 1. Length priority + sub-expression elimination (non-redundant)
        items = self.remove_subtree_redundancy(items)

        # 2. Take top p%, ensure no less than top_k_common
        p = self.config.quality_top_percent
        cutoff = max(self.config.top_k_common, int(len(items) * p))
        candidates = items[:cutoff]

        # 3. Sort by frequency in descending order
        candidates.sort(key=lambda x: -x[2])

        # 4. Take top-k
        top_items = candidates[:self.config.top_k_common]

        return [list(k) for k, _, _ in top_items]

    def remove_subtree_redundancy(self, items):
        """Sort by length descending + quality ascending, eliminate candidates that are subtrees of earlier expressions"""
        if len(items) <= 1:
            return items

        # ✅ Composite sort: descending by length, then ascending by avg_fitness (better quality first)
        sorted_items = sorted(items, key=lambda x: (-len(x[0]), x[1]))
        result = []

        for i, (expr, avg_fit, freq) in enumerate(sorted_items):
            is_subtree = False
            for j in range(i):
                ancestor_expr = sorted_items[j][0]
                if self.is_subtree_of(tuple(expr), tuple(ancestor_expr)):
                    is_subtree = True
                    break
            if not is_subtree:
                result.append((expr, avg_fit, freq))

        return result

    def is_subtree_of(self, sub_key: tuple, tree_key: tuple) -> bool:
        """Check if sub_key is a subtree of tree_key (based on token sequence matching)"""
        sub_list = list(sub_key)
        tree_list = list(tree_key)
        n, m = len(tree_list), len(sub_list)
        for i in range(n - m + 1):
            if tree_list[i:i+m] == sub_list:
                return True
        return False

    # -------- Safe subtree construction and replacement --------
    def safe_subtree_from_prefix(self, prefix_expr: List[str]):
        try:
            internal_str = prefix_to_internal_expression(prefix_expr, self.operator_functions)
            subtree = gp.PrimitiveTree.from_string(internal_str, self.sr.pset)
            return subtree
        except Exception:
            return None

    def is_valid_subtree_replacement(self, ind, point: int, subtree) -> bool:
        try:
            test_ind = creator.Individual(self.sr.toolbox.clone(ind))
            test_ind[point:point+1] = subtree
            if test_ind.height > self.config.gp_params['max_height']:
                return False
            if len(test_ind) > self.config.gp_params['max_size']:
                return False
            return self.validator.is_valid(test_ind)
        except Exception:
            return False

    # -------- Structured mutation (calls safe replacement) --------
    def structured_mutate_individual(self, ind: creator.Individual, mask_set: set) -> creator.Individual:
        base_ind = creator.Individual(self.sr.toolbox.clone(ind))
        working = creator.Individual(self.sr.toolbox.clone(ind))

        util_genes = self.extract_common_subexpressions_from_global_hof() if self.current_round > 0 else []
        explore_size = min(self.config.max_explore_batch, self.exploration_manager.get_unexplored_count())
        explore_genes = self.exploration_manager.sample_batch_with_mask(
            explore_size,
            mask_set=mask_set,
            mark_as_seen=True
        ) if explore_size > 0 else []

        if not util_genes:
            return base_ind

        max_util = min(self.config.fuse_n, len(util_genes))
        util_k = random.randint(1, max_util)
        max_explore = min(self.config.fuse_m, len(explore_genes))
        explore_k = random.randint(0, max_explore) if max_explore > 0 else 0

        chosen_utils = random.sample(util_genes, util_k) if util_k > 0 else []
        chosen_explores = random.sample(explore_genes, explore_k) if explore_k > 0 else []
        inject_list = [('util', g) for g in chosen_utils] + [('explore', g) for g in chosen_explores]
        random.shuffle(inject_list)

        for _, gene in inject_list:
            if len(working) < 2:
                break
            subtree = self.safe_subtree_from_prefix(gene)
            if subtree is None:
                continue
            point = random.randint(0, len(working) - 1)
            if not self.is_valid_subtree_replacement(working, point, subtree):
                continue
            working[point:point+1] = subtree

        if self.validator.is_valid(working) and len(working) <= self.config.gp_params['max_size']:
            return working
        else:
            return base_ind

    # -------- Initial population construction --------
    def build_initial_population_with_gene_injection(self, mask_set: set) -> list:
        C = self.config.gp_params["pops"]
        population = []

        for _ in range(C):
            ind = self.sr.toolbox.individual()
            if random.random() < self.config.inject_prob:
                try:
                    mutated = self.structured_mutate_individual(ind, mask_set)
                    if self.validator.is_valid(mutated) and len(mutated) <= self.config.gp_params['max_size']:
                        ind = mutated
                except Exception:
                    pass
            population.append(ind)

        return population

    # -------- Build exploration mask --------
    def build_exploration_mask(self) -> set:
        """Build Exploration mask for the next round: mask global + previous round local HOF sub-expressions"""
        mask_set = set()
        seen_exprs = set()

        # 1. Global HOF
        for fitness, ind in self.global_hof:
            expr_str = str(ind)
            if expr_str in seen_exprs:
                continue
            seen_exprs.add(expr_str)
            try:
                prefix_seq = expr_to_sequence(expr_str)
                subexpr_tuples = self.subexpr_extractor.extract(
                    [prefix_seq],
                    self.config.explore_min_L,
                    self.config.explore_max_L
                )
                for t in subexpr_tuples:
                    mask_set.add(tuple(t))
            except Exception:
                continue

        # 2. Previous round local HOF (self.current_hof_items)
        for ind in getattr(self, 'current_hof_items', []):
            expr_str = str(ind)
            if expr_str in seen_exprs:
                continue
            seen_exprs.add(expr_str)
            try:
                prefix_seq = expr_to_sequence(expr_str)
                subexpr_tuples = self.subexpr_extractor.extract(
                    [prefix_seq],
                    self.config.explore_min_L,
                    self.config.explore_max_L
                )
                for t in subexpr_tuples:
                    mask_set.add(tuple(t))
            except Exception:
                continue

        return mask_set

    # -------- Frequency Decay --------
    def apply_exploration_decay(self):
        """Product decay for exploration frequency"""
        decay = self.config.explore_decay_factor
        for key in self.exploration_manager.frequency:
            self.exploration_manager.frequency[key] *= decay

    # -------- Main Execution Flow --------
    def run(self, X: np.ndarray, y: np.ndarray, on_round_end=None):
        self.X = X
        self.y = y
        self.sr.set_data(X, y)

        best_result = None
        self.current_hof_items = []  # Initialize

        for round_idx in range(self.config.num_rounds):
            self.current_round = round_idx
            # --- Check if top-k changed ---
            round_topk = self.extract_common_subexpressions_from_global_hof()
            topk_key = tuple(tuple(expr) for expr in sorted(round_topk))

            if self.last_topk is not None:
                if topk_key == self.last_topk:
                    self.stable_count += 1
                    #print(f"🔄 Consensus stable count: {self.stable_count}")
                else:
                    self.stable_count = 0
            self.last_topk = topk_key

            # --- Check if restart is needed ---
            if self.stable_count >= self.restart_patience:
                #print(f"🔥 Round {round_idx} round: Consensus stagnation detected, restarting global HOF")
                # Keep the best individual, clear the rest
                if self.global_hof:
                    best_ind = min(self.global_hof, key=lambda x: x[0])
                    self.global_hof = [best_ind]  # Keep best
                else:
                    self.global_hof = []
                    self.stable_count = 0  # Reset count
                # Optional: reset exploration frequency
                # self.exploration_manager.frequency.clear()

            # --- Build initial population ---
            if round_idx == 0:
                initial_population = [self.sr.toolbox.individual() for _ in range(self.config.initial_random_size)]
            else:
                mask_set = self.build_exploration_mask()
                initial_population = self.build_initial_population_with_gene_injection(mask_set)

            # --- Run GP ---
            best_ind, best_fit, hof_items = self.sr.ga_run(initial_pop=initial_population)

            # --- Save this round's hof_items for next round use ---
            self.current_hof_items = hof_items

            # --- Update global HOF ---
            for ind in [best_ind] + list(hof_items):
                fitness = ind.fitness.values[0]
                if np.isfinite(fitness) and fitness < 1e6:
                    expr_str = str(ind)
                    if expr_str not in self.seen_global_exprs:
                        self.seen_global_exprs.add(expr_str)
                        self.global_hof.append((fitness, ind))
            self.global_hof = sorted(self.global_hof, key=lambda x: x[0])[: self.max_global_hof_size]

            # --- High-frequency sub-expressions for this round ---
            round_topk = self.extract_common_subexpressions_from_global_hof()

            # --- Frequency product decay ---
            self.apply_exploration_decay()

            # --- Save history & callback ---
            round_info = {
                "round": round_idx,
                "best_fitness": best_fit,
                "best_expr_str": str(best_ind),
                "unexplored_count": self.exploration_manager.get_unexplored_count(),
                "global_hof_size": len(self.global_hof),
                "topk_common_subexpr": round_topk,
            }
            self.history.append(round_info)

            if on_round_end is not None:
                try:
                    on_round_end(round_info)
                except Exception:
                    pass

            # --- Early termination ---
            threshold = self.config.gp_params['threshold']
            if best_fit < threshold:
                best_result = {
                    "round": round_idx,
                    "best_fitness": best_fit,
                    "best_expr_str": str(best_ind),
                    "early_stopped": True
                }
                break

            # --- Save HOF ---
            self.save_hall_of_fame(round_idx, hof_items)

        # --- End processing ---
        if best_result is None:
            best = min(self.history, key=lambda x: x["best_fitness"])
            best_result = {
                "round": best["round"],
                "best_fitness": best["best_fitness"],
                "best_expr_str": best["best_expr_str"],
                "early_stopped": False
            }

        return best_result

    def save_hall_of_fame(self, round_idx: int, hof_items: list):
        filepath = os.path.join(self.config.output_dir, f"hall_of_fame_round_{round_idx}.txt")
        with open(filepath, "w") as f:
            f.write(f"Round {round_idx + 1}\n")
            f.write(f"Population Size: {self.config.gp_params['pops']}\n")
            f.write("Hall of Fame:\n")
            for i, ind in enumerate(hof_items):
                f.write(f"{i+1}: {ind} | Fitness: {ind.fitness.values[0]:.6e}\n")


# =========================
# Example main program (printed only here)
# =========================
def main():
    random.seed(0)

    for i in range(10):
        np.random.seed(i)

        config = Config()

        X = np.random.uniform(0, 10, (config.n_samples, config.input_dim))
        x1 = X[:, 0]
        x2 = X[:, 1]
        y_true = x1*x1*x1*x1 - x1*x1*x1 + 0.5*x2*x2 - x2
        y = y_true

        def progress_print(info):
            r = info["round"] + 1
            print(f"Round {r} | Best MSE = {info['best_fitness']:.6e}")
            print(f"Best Expression: {info['best_expr_str']}")
            if info["topk_common_subexpr"]:
                pretty = [ " ".join(tokens) for tokens in info["topk_common_subexpr"] ]
                print("Top-k frequent subexpressions in this round:")
                for i, s in enumerate(pretty, 1):
                    print(f"  {i}. {s}")
            else:
                print("Top-k frequent subexpressions in this round:None")
            print()

        rgp = ReinforcementGeneticProgramming(config)
        result = rgp.run(X, y, on_round_end=progress_print)

        print("🎉 ========== Run Completed ==========")
        print(f"Best Round: Round {result['round'] + 1}" + (" (Early terminated)" if result.get('early_stopped') else ""))
        print(f"Best MSE: {result['best_fitness']:.6e}")
        print(f"Best Expression:\n{result['best_expr_str']}")

        with open("sr_history.pkl", "wb") as f:
            pickle.dump(rgp.history, f)
        print("📊 History saved to sr_history.pkl")


if __name__ == "__main__":
    main()