import numpy as np
from solutions import Solution

def solution_dominates(sol_A: Solution, sol_B: Solution) -> bool:
    """
    Checks if sol_A dominates sol_B.
    - A solution A dominates B if:
    1. A is feasible and B is not.
    2. Both are feasible and A is no worse in all objectives and better in at least one.
    3. Both are infeasible and A has a lower total constraint violation than B.
    """
    v_A = np.sum(np.maximum(0, sol_A.constraints))
    v_B = np.sum(np.maximum(0, sol_B.constraints))

    if v_A > 0 and v_B == 0:
        return False
    if v_A == 0 and v_B > 0:
        return True 
    if v_A > 0 and v_B > 0:
        # Both infeasible: lower violation wins
        return v_A < v_B

    a_obj = sol_A.objectives
    b_obj = sol_B.objectives
    
    if np.any(a_obj > b_obj):
        return False 
        
    if np.any(a_obj < b_obj):
        return True 
        
    return False # Both are equal