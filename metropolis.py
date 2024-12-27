import numpy as np
from numba import jit

@jit(nopython=True)
def metropolis_criteria(delta_E, T):
    """
    Description
    -----------
    Implementation of the Metropolis algorithm.

    Parameters
    ----------
    delta_E: float
        Change in lattice energy by switching to a new configuration (flipping a spin)

    T: float
        Temperature of the lattice

    Returns
    -------
    -1 (int) if spin should be flipped
    1 (int) if spin should not be flipped
    """
    if delta_E <= 0:
        return -1
    elif np.random.random() <= np.exp(-delta_E / T):
        return -1
    else:
        return 1

@jit(nopython=True)
def step(L, lattice, T):
    """
    Description
    -----------
    Function to execute a step within a simulation sweep.
    A single step consists of applying the Metropolis algorithm to a single spin site

    Parameters
    ----------
    L: int
        Lattice dimension
    lattice: numpy ndarray of ints
        LxL numpy array storing the spin configuration of the Ising model
    T: float
        Temperature of lattice

    Returns: void
    """
    for x in range(L):
        for y in range(L):
            delta_E = lattice.delta_E_at_site(x, y)
            if metropolis_criteria(delta_E, T) == - 1:
                lattice.flip_spin(x, y)
    return