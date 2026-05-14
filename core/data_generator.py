# data_generator.py
"""
Generate data for symbolic regression tasks based on dataset.csv
Use sympy for symbolic parsing and numerical computation
Support user-provided random seed
"""

import pandas as pd
import numpy as np
import sympy as sp
from typing import Tuple, Dict, Any

def generate_data_by_name(
    name: str,
    csv_file: str = "dataset.csv",
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Generate data based on expression name

    Args:
        name (str): Expression name, e.g., "Nguyen-12"
        csv_file (str): dataset.csv file path
        seed (int, optional): Random seed used for data generation

    Returns:
        X (np.ndarray): Input data, shape (n_samples, n_features)
        y (np.ndarray): Output data, shape (n_samples,)
        expr_str (str): True expression string
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        # Note: DEAP uses random module, but data generation uses numpy
        # For full reproducibility, random.seed(seed) should also be set externally

    # Read dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Find corresponding row
    row = df[df['name'] == name]
    if row.empty:
        available = df['name'].tolist()
        raise ValueError(f"Expression not found: {name}\nAvailable expressions: {available}")

    expr_str = row['expression'].values[0].strip()
    var_ranges_str = row['variable_ranges'].values[0].strip()
    n_samples = int(row['B'].values[0])

    # Parse variable ranges
    try:
        var_ranges = eval(var_ranges_str)  # {'x': [-1,1], 'y': [0,1]}
        if not isinstance(var_ranges, dict):
            raise ValueError("variable_ranges must be a dictionary")
    except Exception as e:
        raise ValueError(f"Failed to parse variable_ranges: {e}")

    variables = list(var_ranges.keys())
    bounds = [var_ranges[var] for var in variables]

    # Use sympy to Parse expression
    try:
        # Define symbolic variables
        symbols = {var: sp.Symbol(var) for var in variables}
        # Parse string expression
        expr_sympy = sp.sympify(expr_str, locals=symbols)
        # Convert to callable numerical function
        func = sp.lambdify(symbols.values(), expr_sympy, 'numpy')
    except Exception as e:
        raise RuntimeError(f"Parse expression '{expr_str}' error: {e}")

    # Generate random data
    X = np.zeros((n_samples, len(variables)))
    for i, (var, (low, high)) in enumerate(var_ranges.items()):
        X[:, i] = np.random.uniform(low, high, n_samples)

    # Calculate y
    try:
        if len(variables) == 1:
            y = func(X[:, 0])
        else:
            y = func(*[X[:, i] for i in range(len(variables))])
        # Ensure y is a 1D array
        y = np.asarray(y).flatten()
    except Exception as e:
        raise RuntimeError(f"Error calculating y: {e}")

    return X, y, str(expr_sympy)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Example 1: No seed specified
    print("=== Example 1: No seed specified ===")
    X1, y1, expr1 = generate_data_by_name("Nguyen-12", "dataset/dataset.csv")
    print(f"Expression: {expr1}")
    print(f"X shape: {X1.shape}")
    print(f"y shape: {y1.shape}")
    print(f"First 3 samples:")
    for i in range(3):
        print(f"  x={X1[i,0]:.3f}, y={X1[i,1]:.3f} -> y={y1[i]:.3f}")

    print("\n=== Example 2: Specified seed ===")
    # Example 2: Specified seed to ensure reproducibility
    X2a, y2a, expr2 = generate_data_by_name("Nguyen-12", "dataset/dataset.csv", seed=42)
    X2b, y2b, _ = generate_data_by_name("Nguyen-12", "dataset/dataset.csv", seed=42)
    print(f"Are the two generated datasets identical: {np.allclose(X2a, X2b) and np.allclose(y2a, y2b)}")

    print("\n=== Example 3: Error handling ===")
    try:
        generate_data_by_name("Nonexistent-Name", "dataset/dataset.csv")
    except ValueError as e:
        print(f"Caught expected error: {e}")