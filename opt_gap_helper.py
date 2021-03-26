import autograd.numpy as np
from pymanopt.manifolds.sphere import Sphere
from matrix_clr import *
from matplotlib import pyplot as plt


def get_neighbors_submatrix(contrib_mat: np.ndarray,
                            trust_mat: np.ndarray,
                            grant: int) -> np.ndarray:
    """
    returns the matrix corresponding only to contributions and users
    in the NeighborsSubgraph of a grant (the vertices which are within 
    distance 3 of the original grant)
   
    Args:
        contrib_mat: the original matrix of users and grants
        trust_mat: original trust matrices
        grant: the grant number to return
      
    Returns:
       adj_submatrix: a new matrix consisting of users who gave 
       to the grant, the other grants that these users gave to, 
       and the other users who gave to these other grants
       
       trust_submatrix: a submatrix of trust values corresponding to
       original users
   """
    all_users = np.arange(contrib_mat.shape[0])
    all_grants = np.arange(contrib_mat.shape[1])
    original_users = all_users[contrib_mat[:, grant] > 0]  # dist 1 to grant
    dist1 = contrib_mat[original_users, :]
    # distance 1 = all users who contributed to original
    check_new_grants_from_users = np.sum(dist1, axis=0) > 0
    # distance 2 = collects any grant that has any contribution from any user
    # that contributed to original grant (including original grant)
    final_grants = all_grants[check_new_grants_from_users]
    dist2 = contrib_mat[:, final_grants]
    check_new_users_from_grants = np.sum(dist2, axis=1) > 0
    final_users = all_users[check_new_users_from_grants]
    nabe_indices = tuple([final_users[:, None], final_grants])
    trust_indices = tuple([final_users[:, None], final_users])
    neighbors_submat = contrib_mat[nabe_indices]
    trust_submat = trust_mat[trust_indices]
    return neighbors_submat, trust_submat


def get_conj_optimal(contrib_mat: np.ndarray) -> float:
    """
    returns the conjectured optimal amount corresponding to a contribution
    matrix (notice this will primarily be called on a **submatrix** given by 
    the **get_neighbors_submatrix** command)
    
    Args:
        contrib_mat: a matrix of contributions
        
    Returns:
        optimal: the amount (users)*(users-1)*budget/(2*(users+budget))
    """
    n_users = contrib_mat.shape[0]
    budget = np.linalg.norm(contrib_mat)
    optim_numerator = (n_users) * (n_users - 1) * budget
    optim_denominator = 2 * (n_users + budget)
    optim = optim_numerator / optim_denominator
    return optim


def optimality_gap(contrib_mat: np.ndarray,
                   trust_mat: np.ndarray,
                   grant: int) -> float:
    """
    For a given contribution matrix, find the optimality gap
    for a given grant.
   
    Args:
        contrib_mat: a matrix where the (i,j) entry is sqrt
                   of user i contribution to grant j
        grant: the grant number to investigate
   """

    nabe_submat, trust_submat = get_neighbors_submatrix(contrib_mat,
                                                        trust_mat,
                                                        grant)
    nabe_submat_grants = nabe_submat.shape[1]
    grant_nums = range(nabe_submat_grants)
    nabe_submat_actual_funds = pairwise_qf_calc_grants(nabe_submat,
                                                       trust_submat,
                                                       grant_nums)
    nabe_submat_act_total = np.sum(nabe_submat_actual_funds)
    nabe_submat_conj_opt = get_conj_optimal(nabe_submat)
    opt_gap = 1.0 - (nabe_submat_act_total / nabe_submat_conj_opt)
    return opt_gap


def opt_gap_for_rand_alloc(n_users: int, k_grants: int,
                           tot_funds: float,
                           trust_mat: np.ndarray) -> np.ndarray:
    """
    assumes n_users have tot_funds to contribute to k_grants
    randomly allocates the funds to the grants
    returns optimality gap for each grant
    
    Args:
       n_users: number of users
       k_grants: number of grants
       tot_funds: total funds that the users have to allocate
       trust_matrix: a matrix of trust users 
       
    Returns:
       optimality_gaps: an array of each grant's optimality gap
    """
    alloc_sphere = Sphere(n_users, k_grants)  # construct sphere of allocations
    rand_alloc = np.abs(alloc_sphere.rand())
    optimality_gaps = [optimality_gap(rand_alloc, trust_mat, grant)
                       for grant in range(k_grants)]
    optimality_gaps_array = np.array(optimality_gaps)
    return optimality_gaps_array


def opt_gap_experiment(n_users: int, k_grants: int,
                       tot_funds: float,
                       trust_matrix: np.ndarray,
                       num_trials: int):
    experiments = np.array([opt_gap_for_rand_alloc(n_users,
                                                   k_grants,
                                                   tot_funds,
                                                   trust_matrix)
                            for trial in range(num_trials)])
    return experiments
