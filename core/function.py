# function.py

import operator
import numpy as np
from typing import List, Dict, Callable, Tuple
from deap import gp
import re
from numba import njit

# =========================
# Numerically stable basic operators (supports numpy batching)
# =========================
_EPS = 1e-12


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / (b + _EPS)


def sin(a):
    return np.sin(a)


def cos(a):
    return np.cos(a)


def exp(a):
    return np.exp(np.clip(a, -50, 50))


def log(a):
    return np.log(np.abs(a) + _EPS)


# =========================
# Numba accelerated MSE (vectorized)
# =========================
@njit(fastmath=True)
def mse_numba(pred: np.ndarray, y_true: np.ndarray) -> float:
    diff = pred - y_true
    return np.mean(diff * diff)


# =========================
# Register operators
# =========================
def register_operator_functions() -> Dict[str, Dict]:
    """Return operator dictionary (used for constraints)"""
    operator_functions = {
        '+': {'function': add, 'arity': 2},
        '-': {'function': sub, 'arity': 2},
        '*': {'function': mul, 'arity': 2},
        '/': {'function': div, 'arity': 2},
        'sin': {'function': sin, 'arity': 1},
        'cos': {'function': cos, 'arity': 1},
        'exp': {'function': exp, 'arity': 1},
        'log': {'function': log, 'arity': 1},
    }
    return operator_functions


def register_operators(pset: gp.PrimitiveSet, operator_functions: dict) -> None:
    for _, op_info in operator_functions.items():
        func: Callable = op_info["function"]
        num_children: int = op_info["arity"]
        pset.addPrimitive(func, num_children)


# =========================
# Expression conversion tools
# =========================
def prefix_to_internal_expression(prefix_expr: List[str], operator_functions: dict) -> str:
    """
    Convert preorder expression (e.g., ["+", "x_2", "sin", "x_1"]) to function string (e.g., "add(x_2, sin(x_1))")
    """
    stack = []
    op2func = {k: v["function"].__name__ for k, v in operator_functions.items()}
    for token in reversed(prefix_expr):
        if token in operator_functions:
            ar = operator_functions[token]["arity"]
            children = [stack.pop() for _ in range(ar)]
            expr = f"{op2func[token]}({', '.join(children)})"
            stack.append(expr)
        else:
            stack.append(token)
    return stack[0]


# function.py

def expr_to_sequence(expr: str) -> List[str]:
    """Parse "add(add(x_1,x_2),x_3)" into ["+", "+", "x_1", "x_2", "x_3"]"""
    # Define mapping from function names to symbols
    func2op = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'sin': 'sin',
        'cos': 'cos',
        'exp': 'exp',
        'log': 'log'
    }
    tokens = re.findall(r"[\w]+|[\+\-\*/(),]", expr)
    stack = []
    result = []

    for token in tokens:
        if token == '(':
            stack.append(result)
            result = []
        elif token == ')':
            func = stack.pop()
            func.append(result)
            result = func
        elif token == ',':
            continue
        else:
            # Convert function names to operators
            op = func2op.get(token, token)
            result.append(op)

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
    return list(flatten(result))


# =========================
# Syntax tree-based expression legality check (structured)
# =========================
class ExpressionValidator:
    def __init__(self, operator_functions):
        self.op_names = {v["function"].__name__: k for k, v in operator_functions.items()}
        self.trig_ops = {'sin', 'cos'}
        self.exp_log_ops = {'exp', 'log'}

    def is_valid(self, individual: gp.PrimitiveTree) -> bool:
        try:
            return self._check_tree(individual)
        except:
            return False

    def _check_tree(self, tree: gp.PrimitiveTree) -> bool:
        # Use stack to simulate recursion, record parent call chain (keep only function names)
        stack = []  # Store (node, parent_chain), parent_chain is list of ancestor function names

        for node in tree:
            if isinstance(node, gp.Primitive):
                name = self.op_names[node.name]
                # Current node is a function
                current_chain = [name]
                if stack:
                    current_chain = stack[-1][1] + [name]
                stack.append((node, current_chain))
                # Check if current chain violates rules
                if not self._is_chain_valid(current_chain):
                    return False
            elif isinstance(node, gp.Terminal):
                # Terminals (variables) do not change chain, but need to pop previous function's stack frame
                if stack and isinstance(stack[-1][0], gp.Primitive):
                    stack.pop()
        return True

    def _is_chain_valid(self, chain: List[str]) -> bool:
        if len(chain) < 2:
            return True

        top = chain[-1]   # Current function
        parent = chain[-2]  # Direct parent function

        # 1. Prohibit trigonometric function nesting
        if top in self.trig_ops and parent in self.trig_ops:
            return False

        # 2. Prohibit log and trig mutual nesting
        if top in self.trig_ops and parent == 'log':
            return False
        if top == 'log' and parent in self.trig_ops:
            return False

        # 3. Prohibit exp/log mutual or self-nesting
        if top in self.exp_log_ops and parent in self.exp_log_ops:
            return False

        return True