"""Optimizing k-fold splits with groups.

These functions aid strat_k_folds.py in calculating the
optimal fold allocation for all the groups in the dataset
ensuring the folds are as equal in size as they can be,
while also being as close to the actual class distribution
as possible.

Downloaded and modified from:
https://github.com/joaofig/strat-group-split/tree/main
"""
from typing import Set, Tuple
import numpy as np
from numpy.random import default_rng
from numba import njit


@njit
def calculate_cost(problem: np.ndarray,
                   solution: np.ndarray,
                   k: int) -> float:
    """Calculate difference of current solution to optimal solution.

    Args:
        problem (np.array): A matrix with a column per class and the
            class counts for each group as the values.
        solution (np.ndarray): A 1D array where each value is the current
            fold allocation for the corresponding group.
        k (int): Number of folds.

    Returns:
        float: The summation of the differences between the folds'
            class distributions from the optimal class distribution, and the
            size of the folds to the size the folds should be.
    """
    cost = 0.0
    total = np.sum(problem)
    class_sums = np.sum(problem, axis=0)
    num_classes = problem.shape[1]

    for i in range(k):
        idx = solution == i
        fold_sum = np.sum(problem[idx, :])

        # Start by calculating the fold imbalance cost
        cost += (fold_sum / total - 1.0 / k) ** 2

        # Now calculate the cost associated with the class imbalances
        # Katie: had to add division by 0 error for if fold_sums equal 0
        # there were no chick begging calls during test so this row was 0
        for j in range(num_classes):
            if fold_sum == 0:
                cost += (0 - class_sums[j] / total) ** 2
            else:
                sum_problem = np.sum(problem[idx, j]) / fold_sum
                cost += (sum_problem - class_sums[j] / total) ** 2
    return cost


@njit
def generate_search_space(problem: np.ndarray,
                          solution: np.ndarray,
                          k: int) -> np.ndarray:
    """Generate the search space.

    Args:
        problem (np.ndarray): A matrix with a column per class and the
            class counts for each group as the values.
        solution (np.ndarray): The last known solution.
        k (int): Number of folds.

    Returns:
        np.ndarray: The search space. Folds as columns and cost values
            for each group with a placeholder in one fold each to allow for
            a cost calculation relative to the placeholder.
    """
    num_groups = problem.shape[0]

    space = np.zeros((num_groups, k))
    sol = solution.copy()

    for i in range(num_groups):
        for j in range(k):
            if solution[i] == j:
                space[i, j] = np.inf
            else:
                sol[i] = j
                space[i, j] = calculate_cost(problem, sol, k)
        sol[i] = solution[i]
    return space


@njit
def solution_to_str(solution: np.ndarray) -> str:
    """Convert the solution to a string.

    Args:
        solution (np.ndarray): The current solution.
    Returns:
        str: The current solution as a string.
    """
    return "".join([str(n) for n in solution])


def generate_initial_solution(problem: np.ndarray,
                              k: int,
                              algo: str = "k-bound") -> np.ndarray:
    """Generate the first solution.

    Args:
        problem (np.array): A matrix with a column per class and the
            class counts for each group as the values.
        k (int): The number of folds.
        algo (str): Method for creating initial solution. Defaults to a
            greedy algorithm to satisfy fold proportion requirements only.

    Returns:
        np.ndarray: A 1D array where each value is the current fold
            allocation for the corresponding group.
    """
    num_groups = problem.shape[0]
    if algo == "k-bound":
        rng = default_rng()
        total = np.sum(problem)
        indices = rng.permutation(problem.shape[0])

        solution = np.zeros(num_groups, dtype=int)
        current_fold = 0
        fold_total = 0
        for i in indices:
            group = np.sum(problem[i, :])
            if fold_total + group < total / k:
                fold_total += group
            else:
                current_fold = (current_fold + 1) % k
                fold_total = group
            solution[i] = current_fold
    elif algo == "random":
        rng = default_rng()
        solution = rng.integers(low=0, high=k, size=num_groups)
    elif algo == "zeros":
        solution = np.zeros(num_groups, dtype=int)
    else:
        raise Exception("Invalid algorithm name")
    return solution


def solve(problem: np.ndarray,
          k=5,
          min_cost=1e-5,
          max_retry=100,
          verbose=False) -> np.ndarray:
    """Solve the problem.

    Args:
        problem (np.ndarray):
        k (int): Number of folds, default 5.
        min_cost (float): The largest the cost can be for an
            acceptable solution. Default 1e-5.
        max_retry (int): The max amount of times the program will
            attempt to alter the current solution for a more
            optimal one.
        verbose (bool): True for more debug prints, defaults to False.

    Returns:
        np.ndarray: Optimized solution as a 1D array where each
            value is the fold allocation for each group.
    """
    hist = set()
    retry = 0

    solution = generate_initial_solution(problem, k)
    incumbent = solution.copy()
    low_cost = calculate_cost(problem, solution, k)
    cost = 1.0
    while retry < max_retry and cost > min_cost:
        decision = generate_search_space(problem, solution, k=5)
        grp, cls = select_move(decision, solution, hist)

        if grp != -1:
            solution[grp] = cls
            cost = calculate_cost(problem, solution, k=5)
            if cost < low_cost:
                low_cost = cost
                incumbent = solution.copy()
                retry = 0
                if verbose:
                    print(cost)
            else:
                retry += 1
            hist.add(solution_to_str(solution))
    return incumbent


def select_move(decision: np.ndarray,
                solution: np.ndarray,
                history: Set) -> Tuple:
    """Select the change to make to the current solution.

    Args:
        decision (np.ndarray): The current search space matrix.
        solution (np.ndarray): The current solution.
        history (Set): Previous solutions.
    Returns:
        Tuple: Position in the solution matrix to move a group
            into a different fold.
    """
    candidates = np.argsort(decision, axis=None)

    for candidate in candidates:
        position = np.unravel_index(candidate, decision.shape)
        sol = solution.copy()
        sol[position[0]] = position[1]
        sol_str = solution_to_str(sol)

        if sol_str not in history:
            return position[0], position[1]
    return -1, -1  # No move found!
