# exploration.py
"""
exploration.py

Function:
    1. Generate all valid preorder expressions in [min_L, max_L] based on symbol library;
    2. Mathematical rationality filtering (trig nesting, exp-log adjacent, etc.);
    3. Support frequency-weighted sampling, unexplored/low-frequency expressions are more easily sampled;
    4. Support 'exploitation mask': some expressions are forbidden to sample (already used by exploitation module);
    5. Provide lookup_table and frequency management.

Design Philosophy:
    - From 'Boolean Exploration' -> 'Frequency-driven Exploration'
    - Exploration != one-time, but a 'heat' process
    - Decouple exploitation and exploration: exploited structures are masked in exploration
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Set, Callable
import random
import numpy as np


# ===========================
# Module 1: Generate all valid preorder expressions
# ===========================

def generate_prefix_expressions_in_range(min_L, max_L, operators, operands):
    """
    Generate all valid preorder traversal expressions within length range [min_L, max_L].
    """
    if min_L < 1 or max_L < min_L:
        return []

    dp = defaultdict(list)

    # 1. Length 1: Operands
    for opd in operands:
        dp[1].append([opd])

    # 2. Construct longer expressions
    for length in range(2, max_L + 1):
        for op, arity in operators.items():
            if arity == 1:
                k = length - 1
                if k in dp:
                    for sub in dp[k]:
                        dp[length].append([op] + sub)
            elif arity == 2:
                for i in range(1, length):
                    j = length - 1 - i
                    if j < 1:
                        continue
                    if i in dp and j in dp:
                        for left in dp[i]:
                            for right in dp[j]:
                                dp[length].append([op] + left + right)

    # 3. Aggregate and deduplicate
    seen = set()
    result = []
    for n in range(min_L, max_L + 1):
        for expr in dp[n]:
            key = tuple(expr)
            if key not in seen:
                seen.add(key)
                result.append(expr)

    result.sort(key=lambda x: (len(x), x))
    return result


# ===========================
# Module 2: Expression tree nodes and tree building
# ===========================

class TreeNode:
    """Expression tree node"""
    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        return f"TreeNode({self.val})"


def build_tree_from_prefix(prefix, symbol_arity):
    if not prefix:
        return None, prefix

    token = prefix[0]
    arity = symbol_arity.get(token, 2)
    node = TreeNode(token)

    remaining = prefix[1:]
    for _ in range(arity):
        child, remaining = build_tree_from_prefix(remaining, symbol_arity)
        if child is None:
            raise ValueError(f"Failed to build subtree: {prefix}")
        node.children.append(child)

    return node, remaining


# ===========================
# Module 3: Mathematical Rationality Filter
# ===========================

class MathematicalFilter:
    """
    Mathematical rationality filter, supporting three configurable rules.
    """

    def __init__(self, symbol_arity,
                 trig_nesting=True,
                 exp_log_adjacent=True,
                 exp_log_trig=True):
        self.symbol_arity = symbol_arity
        self.trig_nesting = trig_nesting
        self.exp_log_adjacent = exp_log_adjacent
        self.exp_log_trig = exp_log_trig

        self.trig_ops = {'sin', 'cos', 'tan'}
        self.exp_log_ops = {'exp', 'log'}

    def _has_trig_descendant(self, node: TreeNode) -> bool:
        if node.val in self.trig_ops:
            return True
        return any(self._has_trig_descendant(child) for child in node.children)

    def contains_invalid_structure(self, node: TreeNode) -> bool:
        if node is None or not node.children:
            return False

        # Rule 1: Trigonometric functions cannot contain trig
        if self.trig_nesting and node.val in self.trig_ops:
            for child in node.children:
                if self._has_trig_descendant(child):
                    return True

        # Rule 2: exp and log directly adjacent
        if self.exp_log_adjacent and node.val in self.exp_log_ops:
            for child in node.children:
                if child.val in self.exp_log_ops:
                    return True

        # Rule 3: exp/log cannot act directly on trig
        if self.exp_log_trig and node.val in self.exp_log_ops:
            for child in node.children:
                if child.val in self.trig_ops:
                    return True

        for child in node.children:
            if self.contains_invalid_structure(child):
                return True

        return False

    def is_valid_expression(self, prefix_expr: list) -> bool:
        try:
            root, remaining = build_tree_from_prefix(prefix_expr, self.symbol_arity)
            if len(remaining) != 0:
                return False
            return not self.contains_invalid_structure(root)
        except Exception:
            return False


# ===========================
# Module 4: Exploration Manager (Upgraded)
# ===========================

class ExplorationManager:
    """
    Exploration Manager: Supports frequency-weighted sampling + exploitation mask
    """

    def __init__(self, min_L: int, max_L: int, operators: Dict[str, int], operands: List[str],
                 trig_nesting=True, exp_log_adjacent=True, exp_log_trig=True):
        self.min_L = min_L
        self.max_L = max_L
        self.operators = operators
        self.operands = operands
        self.symbol_arity = operators.copy()
        self.symbol_arity.update({op: 0 for op in operands})

        self.filter = MathematicalFilter(
            symbol_arity=self.symbol_arity,
            trig_nesting=trig_nesting,
            exp_log_adjacent=exp_log_adjacent,
            exp_log_trig=exp_log_trig
        )

        self.all_candidates = []  # All valid expressions
        self.lookup_table = {}    # tuple(expr) -> index
        self.frequency = defaultdict(int)  # tuple(expr) -> sampling frequency

    def generate_candidates(self):
        """Generate and filter all candidate expressions."""
        #print(f"🔧 Starting to generate valid expressions within length range [{self.min_L}, {self.max_L}]...")
        raw_expressions = generate_prefix_expressions_in_range(self.min_L, self.max_L, self.operators, self.operands)
        #print(f"✅ Generated {len(raw_expressions)} grammatically valid expressions")

        #print("🧮 Applying mathematical rationality filter...")
        filtered = [expr for expr in raw_expressions if self.filter.is_valid_expression(expr)]
        #print(f"✅ Remaining {len(filtered)} rational expressions after filtering")

        self.all_candidates = filtered
        self.lookup_table = {tuple(expr): idx for idx, expr in enumerate(self.all_candidates)}
        # frequency initialized to 0
        #print(f"📊 Initialization complete, total {len(self.all_candidates)} explorable expressions.")

    def update_from_set(self, expression_set_B: List[List[str]]):
        """
        Mark expressions in set B as "explored" (i.e., increase frequency to make them less likely to be sampled)
        """
        for expr in expression_set_B:
            key = tuple(expr)
            if key not in self.lookup_table:
                raise ValueError(f"Expression {' '.join(expr)} not in candidate set, cannot update.")
            # Increase frequency to reduce its sampling probability
            self.frequency[key] += 5  # Configurable weight

    def sample_batch(self, batch_size: int, mark_as_seen: bool = True) -> List[List[str]]:
        """
        Sample from unexplored/low-frequency expressions using frequency weighting.
        """
        return self.sample_batch_with_mask(batch_size, mask_set=set(), mark_as_seen=mark_as_seen)

    def sample_batch_with_mask(self, batch_size: int, mask_set: Set[Tuple[str, ...]], mark_as_seen: bool = True) -> List[List[str]]:
        """
        Avoid expressions in mask_set during sampling (exploitation mask)
        """
        candidates = []
        weights = []

        for expr in self.all_candidates:
            key = tuple(expr)
            if key in mask_set:
                continue  # Skip masked expressions
            if self.frequency[key] < 100:  # Limit maximum frequency
                candidates.append(expr)
                weights.append(1.0 / (self.frequency[key] + 1))

        if len(candidates) < batch_size:
            raise ValueError(f"Cannot sample: requires {batch_size} , but candidate set only has {len(candidates)}  (masked or frequency too high)")

        # Normalize weights
        weights = np.array(weights)
        weights /= weights.sum()

        sampled_indices = np.random.choice(len(candidates), size=batch_size, replace=False, p=weights)
        sampled_expressions = [candidates[i] for i in sampled_indices]

        if mark_as_seen:
            for expr in sampled_expressions:
                self.frequency[tuple(expr)] += 1

        return sampled_expressions

    def get_unexplored_count(self) -> int:
        """Return number of expressions with frequency 0"""
        return sum(1 for expr in self.all_candidates if self.frequency[tuple(expr)] == 0)

    def get_total_count(self) -> int:
        """Return total number of expressions"""
        return len(self.all_candidates)


# ===========================
# Module 5: Test Functions
# ===========================

def test_exploration_manager():
    print("🧪 Starting test ExplorationManager...\n")

    operators = {
        '+': 2, '-': 2, '*': 2, '/': 2,
        'sin': 1, 'cos': 1, 'exp': 1, 'log': 1
    }
    operands = ['x_1']

    manager = ExplorationManager(
        min_L=3,
        max_L=5,
        operators=operators,
        operands=operands,
        trig_nesting=True,
        exp_log_adjacent=True,
        exp_log_trig=True
    )
    manager.generate_candidates()

    print(f"\n📋 Total candidates: {manager.get_total_count()}，Unexplored: {manager.get_unexplored_count()}")
    print("Top 10 candidates:")
    for expr in manager.all_candidates[:10]:
        print(f"   {' '.join(expr)}")

    # Simulate exploitation mask
    mask_set = {('sin', 'x_1'), ('exp', 'x_1')}
    print(f"\n🛡️  Exploitation mask: {mask_set}")

    print(f"\n🔄 Sampling 3 (avoiding mask)...")
    try:
        batch = manager.sample_batch_with_mask(3, mask_set=mask_set, mark_as_seen=True)
        print(f"✅ Sampling successful:")
        for expr in batch:
            print(f"   {' '.join(expr)}")
    except ValueError as e:
        print(f"❌ Sampling failed: {e}")

    print(f"\n🔄 Sampling again...")
    try:
        second = manager.sample_batch_with_mask(3, mask_set=mask_set, mark_as_seen=True)
        print(f"✅ Second sampling:")
        for expr in second:
            print(f"   {' '.join(expr)}")
    except ValueError as e:
        print(f"✅ Expected failure: {e}")

    print(f"\n📊 Frequency statistics (Top 10):")
    for expr in manager.all_candidates[:10]:
        key = tuple(expr)
        print(f"   {' '.join(expr)} -> {manager.frequency[key]}")


if __name__ == "__main__":
    test_exploration_manager()