import autograd.numpy as np


# IMPORTANT NOTE:
# All of the functions below assume that cont_mat
# consists of square roots of the relevant
# contributions.

def pairwise_coord(cont_mat: np.ndarray, i: int, j: int) -> float:
    """
    Calculates the pairwise coordination coefficient for 
    users i and j based on their donation history.

    Args:
        cont_mat: the contribution matrix, num_users-by-num_grants with the (user,grant) entry = contribution from user to grant
        i: user number
        j: user number 

    Returns:
        The pairwise coordination coefficient. 
        (Description/reality check:
        Should be 1.0 if users have no grants in common.
        Should grow smaller as users have more grants 
        in common.) 
    """
    coord = np.dot(cont_mat[i, :], cont_mat[j, :])
    penalty = 1.0 / (1.0 + coord)
    return penalty


def calc_pairwise_coord_mat(cont_mat: np.ndarray) -> np.ndarray:
    """
    Returns an upper triangular matrix T where T[i,j] 
    gives the pairwise coordination penalty for users i
    and users j (assuming i < j). 

    Args: 
        cont_mat: A matrix of user contributions to grants.

    Returns:
        coord_penalty_mat: An upper triangular matrix T where
                           T[i,j] is the pairwise coordination
                           coefficient for users i and j
    """
    num_users = cont_mat.shape[0]
    calc_pair_coord = lambda i, j: (i < j) * pairwise_coord(cont_mat, i, j)
    coord_penalty_mat = np.fromfunction(np.vectorize(calc_pair_coord),
                                        shape=(num_users, num_users),
                                        dtype=int)
    return coord_penalty_mat


def pairwise_term(cont_mat: np.ndarray, grant: int, i: int, j: int) -> float:
    """
    gives the product of (i donation to grant) and 
    (j donation to grant) from a contributions matrix

    Args:
        cont_mat: a matrix of contributions 
        grant: the grant under consideration
        i: user number
        j: user number

    Returns:
        the product of the relevant matrix entries 

    """
    term = cont_mat[i, grant] * cont_mat[j, grant]
    return term


def calc_pairwise_term_mat(cont_mat: np.ndarray, grant: int) -> np.ndarray:
    """
    Calculates the matrix of pairwise terms.

    Args:
       cont_mat: The matrix of contributions
       grant: The grant under consideration

    Returns:
       An upper-triangular matrix where the (i,j) entry is
       (i donation to grant) * (j donation to grant)
    """
    num_users = cont_mat.shape[0]
    calc_pair_term = lambda i, j: (i < j) * pairwise_term(cont_mat,
                                                          grant, i, j)
    match_mat = np.fromfunction(np.vectorize(calc_pair_term),
                                shape=(num_users, num_users),
                                dtype=int)
    return match_mat


def pairwise_qf_calc(cont_mat: np.ndarray, trust_mat: np.ndarray,
                     grant: int) -> float:
    """
    Calculates the quadratic funding match with pairwise coordination coefficient.

    Args:
        cont_mat: The matrix of contributions from users to grants.
        trust_mat: The trust score for each pair of users.
        grant: The grant number

    Returns:
       qf_match: The quadratic funding match due to this grant
    """
    num_users = trust_mat.shape[0]
    pair_match_mat = calc_pairwise_term_mat(cont_mat, grant)
    pair_penalty_mat = calc_pairwise_coord_mat(cont_mat)
    qf_match = np.sum(trust_mat * pair_match_mat * pair_penalty_mat)
    return qf_match


def pairwise_qf_calc_grants(cont_mat: np.ndarray, trust_mat: np.ndarray, grant_nums: list) -> np.ndarray:
    """
    Calculates the pairwise qf scores (matches in case 
    of unlimited matching pool) for a set of grant numbers

    Args:
        cont_mat: The contribution matrix
        trust_mat: Trust bonus for each pair of users
        grant_nums: A list of grant numbers

    Returns:
       scores: An array of the scores for each list number

    """

    scores = np.array([pairwise_qf_calc(cont_mat, trust_mat, num)
                       for num in grant_nums])
    return scores


def pairwise_qf_grant_allocations(cont_mat: np.ndarray, trust_mat: np.ndarray, match_pool: float) -> np.ndarray:
    """
    Gives the grant allocations for all grants, 
    proportional to (score for grant)/(all grant scores),
    times the match_pool amount.

    Args:
       cont_mat: A (users)-by-(grants) matrix of contributions
       trust_mat: A (users)-by-(users) matrix giving trust for each pair
       match_pool: The total amount of matching funds available

    Returns:
        an array of the grant allocations (should total to match_pool)
       
    """
    grant_nums = range(cont_mat.shape[1])
    scores = pairwise_qf_calc_grants(cont_mat, trust_mat, grant_nums)
    tot_scores = np.sum(scores)
    matched_proportional_scores = np.array(match_pool
                                           * (1.0 / tot_scores) * scores)
    return matched_proportional_scores
