# gp.py

import random
from typing import List, Tuple
from deap import gp, algorithms, tools, creator, base
import numpy as np
import json

# Import custom function modules
from core.function import (
    register_operator_functions,
    ExpressionValidator,
    prefix_to_internal_expression,
    mse_numba
)

# =========================
# Configuration class
# =========================
class GP_config:
    def __init__(self, config_data):
        self.gp = {
            'pops':       config_data.get("pops", 500),
            'cxpb':       config_data.get("cxpb", 0.5),
            'mutpb':      config_data.get("mutpb", 0.5),
            'times':      config_data.get("times", 100),
            'tournsize':  config_data.get("tournsize", 3),
            'max_height': config_data.get("max_height", 10),
            'max_size':   config_data.get("max_size", 60),
            'hof_size':   config_data.get("hof_size", 5),
            'threshold':  config_data.get("threshold", 0.01),
        }


# =========================
# Symbolic Regression (Genetic Programming)
# =========================
class SymbolicRegression:
    def __init__(self, config_s: GP_config, operator_functions: dict):
        self.operator_functions = operator_functions
        self.config_s = config_s
        self.threshold = config_s.gp['threshold']

        self.data_X = None
        self.data_y_true = None

        # Create Fitness and Individual types (avoid duplicates)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.pset = gp.PrimitiveSet("MAIN", 1)
        self._initialize_primitive_set(1)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=config_s.gp['tournsize'])

        # ✅ Register HallOfFame factory function
        self.toolbox.register("halloffame", tools.HallOfFame, maxsize=config_s.gp['hof_size'])

        # Safe crossover/mutation
        self._register_safe_operators()

        # Initialize fitness cache
        self._fitness_cache = {}

        # Initialize validator
        self.validator = ExpressionValidator(operator_functions)

    def _initialize_primitive_set(self, input_dim: int):
        self.pset = gp.PrimitiveSet("MAIN", input_dim)
        for i in range(input_dim):
            self.pset.renameArguments(**{f"ARG{i}": f"x_{i+1}"})
        # ✅ No constant terminals added anymore
        from core.function import register_operators
        register_operators(self.pset, self.operator_functions)

    # ---------- Safe crossover/mutation ----------
    def _register_safe_operators(self):
        max_h = self.config_s.gp['max_height']
        max_sz = self.config_s.gp['max_size']
        toolbox = self.toolbox

        def within_limits(ind):
            return (ind.height <= max_h) and (len(ind) <= max_sz)

        def _safe_mate(ind1, ind2):
            for _ in range(20):
                c1, c2 = gp.cxOnePoint(toolbox.clone(ind1), toolbox.clone(ind2))
                if within_limits(c1) and within_limits(c2) and \
                   self.validator.is_valid(c1) and self.validator.is_valid(c2):
                    return c1, c2
            return toolbox.clone(ind1), toolbox.clone(ind2)

        def _safe_mutate(ind):
            for _ in range(20):
                c, = gp.mutNodeReplacement(toolbox.clone(ind), pset=self.pset)
                if within_limits(c) and self.validator.is_valid(c):
                    return (c,)
            # Rebirth
            for _ in range(20):
                new_expr = gp.genHalfAndHalf(pset=self.pset, min_=1, max_=2)
                c = creator.Individual(new_expr)
                if within_limits(c) and self.validator.is_valid(c):
                    return (c,)
            return (toolbox.clone(ind),)

        toolbox.register("mate", _safe_mate)
        toolbox.register("mutate", _safe_mutate)

    # ---------- Data ----------
    def set_data(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy.ndarray")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be consistent")

        new_input_dim = X.shape[1]
        self._initialize_primitive_set(new_input_dim)
        self.toolbox.unregister("compile")
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.data_X = X.astype(np.float64)
        self.data_y_true = y.astype(np.float64)

    # ---------- Evaluation ----------
    def evaluate(self, individual):
        key = str(individual)
        if key in self._fitness_cache:
            return (self._fitness_cache[key],)
        try:
            error = self.calculate_fitness(individual)
            self._fitness_cache[key] = float(error)
            return (float(error),)
        except Exception:
            return (float('inf'),)

    def calculate_fitness(self, individual):
        func = self.toolbox.compile(expr=individual)
        args = [self.data_X[:, j] for j in range(self.data_X.shape[1])]
        try:
            pred = func(*args)
        except:
            return float('inf')

        if np.isscalar(pred):
            pred = np.full(self.data_y_true.shape, float(pred), dtype=np.float64)
        else:
            pred = np.asarray(pred, dtype=np.float64)
            if pred.ndim > 1:
                pred = pred.reshape(-1)
            if pred.shape != self.data_y_true.shape:
                return float('inf')

        if not np.all(np.isfinite(pred)):
            return float('inf')

        return float(mse_numba(pred, self.data_y_true))

    # ---------- Run ----------
    def ga_run(self, initial_pop: List = None) -> Tuple[gp.PrimitiveTree, float]:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(self.config_s.gp['hof_size'])

        required_size = self.config_s.gp['pops']



        # Construct initial population (✅ No deduplication)
        if initial_pop is None:
            pop = [self.toolbox.individual() for _ in range(required_size)]
        else:
            # Filter invalid individuals
            initial_pop = [ind for ind in initial_pop if self.validator.is_valid(ind)]
            if len(initial_pop) < required_size:
                additional = [self.toolbox.individual() for _ in range(required_size - len(initial_pop))]
                pop = initial_pop + additional
            else:
                pop = initial_pop[:required_size]  # Truncate

        # Run GA
        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.config_s.gp['cxpb'],
            mutpb=self.config_s.gp['mutpb'],
            ngen=self.config_s.gp['times'],
            stats=stats,
            halloffame=hof,
            verbose=False
        )

        # ✅ HOF Re-evaluation (ensure accuracy)
        best_individual = hof[0]
        best_fitness = self.calculate_fitness(best_individual)  # Recalculate, do not rely on cache
        return best_individual, best_fitness, hof


# =========================
# Example main program
# =========================
def main():
    # 1) Operators
    operator_functions = register_operator_functions()

    # 2) Example data
    np.random.seed(0)
    X = np.random.uniform(-2, 2, size=(60, 3))
    y = 2.0 * X[:, 0] + 3.0 * (X[:, 1] ** 2) - np.sin(X[:, 2])
    X = X.astype(np.float64); y = y.astype(np.float64)

    # 3) Configuration
    config_gp = {
        "pops": 500,
        "times": 80,
        "tournsize": 5,
        "max_height": 10,
        "max_size": 30,
        "hof_size": 5,
        "cxpb": 0.5,
        "mutpb": 0.5,
    }
    config = GP_config(config_gp)

    # 4) Create GP object
    sr = SymbolicRegression(config, operator_functions)
    sr.set_data(X, y)

    # 5) Specify initial population (✅ Retain functionality)
    initial_population_prefix = [
        ["+", "x_2", "sin", "x_1"],
        ["-", "x_2", "x_1"],
        ["*", "x_2", "x_1"],
        ["/", "x_2", "x_1"],
        ["*", "+", "x_2", "x_1", "x_1"],
        ["+", "x_2", "*", "x_1", "x_1"],
    ]

    initial_population = []
    for prefix_expr in initial_population_prefix:
        internal_str = prefix_to_internal_expression(prefix_expr, operator_functions)
        try:
            expr_tree = gp.PrimitiveTree.from_string(internal_str, sr.pset)
            ind = creator.Individual(expr_tree)
            initial_population.append(ind)
        except:
            continue  # Ignore invalid expressions

    # 6) Run GP
    best_individual, best_fitness, hof = sr.ga_run(initial_pop=initial_population)

    # 7) Output results
    print("\n=== Evolution Results ===")
    print("Best expression (function format):")
    print(best_individual)
    print(f"Re-evaluated MSE: {best_fitness:.6f}")

    # —— New: Print Hall of Fame for this run (by rank) ——
    print("\n=== Hall of Fame ===")
    for rank, ind in enumerate(hof, start=1):
        mse = sr.calculate_fitness(ind)  # Re-evaluate individually to ensure consistency
        print(f"[{rank}] Expression: {ind}")
        print(f"    MSE: {mse:.6f}")


if __name__ == "__main__":
    main()