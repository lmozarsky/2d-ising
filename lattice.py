import numpy as np
from numba import int32, float32
from numba.experimental import jitclass

spec = [
    # Size of the lattice
    ('L', int32),       

    # Temperature of the lattice
    ('T', float32),                     

    # 2D array representing the lattice
    ('lattice', int32[:, :])           
]

@jitclass(spec)
class Lattice:
    """
    Lattice class to simulate the 2D Ising model

    Parameters
    ----------
    L: int
        Lattice dimension (L**2 = N, the number of lattice sites)
    T: float
        Lattice temperature
    lattice: numpy ndarray of ints
        LxL numpy array storing the spin configuration of the Ising model
    """

    def __init__(self, L, T):
        """
        Initialize the lattice instance.
        The lattice spin configuration is random, unless the temperature is below 1.0.
        In this case, the lattice is initialized with all spins pointing "up," which 
        helps the lattice converge to a thermalized state.
        """
        self.L = L
        self.T = T
        if self.T < 1.0:
            self.lattice = np.ones((self.L, self.L), dtype=np.int32)
        else:
            self.lattice = np.empty((L, L), dtype=np.int32) 
            for x in range(self.L):
                for y in range(self.L):
                    self.lattice[x, y] = 1 if np.random.rand() < 0.5 else -1

    def get_spin(self, x, y):
        """
        Get the spin of the lattice site at coordinates (x,y)
        """
        return self.lattice[x % self.L, y % self.L]
    
    def flip_spin(self, x, y):
        """
        Flip the spin of the site at (x,y)
        """
        self.lattice[x, y] *= -1

    def sum_neighbors(self, x, y):
        """
        Calculate the sum of the spins of the neighbors of site at (x,y).
        """
        return (
            self.lattice[(x - 1) % self.L, y] +
            self.lattice[(x + 1) % self.L, y] +
            self.lattice[x, (y - 1) % self.L] +
            self.lattice[x, (y + 1) % self.L]
        )

    def delta_E_at_site(self, x, y):
        """
        Compute the configuration energy change if spin at (x,y) is flipped.
        """
        sum_nb = self.sum_neighbors(x, y)
        return 2 * self.get_spin(x, y) * sum_nb

    def get_mag(self):
        """
        Compute the absolute value of the average magnetization per lattice site.
        """
        total = 0
        for x in range(self.L):
            for y in range(self.L):
                total += self.lattice[x, y]
        return np.abs(total)/(self.L * self.L)

    def get_total_energy(self):
        """
        Compute the total energy of the lattice configuration.
        """
        #return total_energy_numba(self.lattice, self.L)
        energy = 0
        for x in range(self.L):
            for y in range(self.L):
                site_value = self.lattice[x,y]
                energy -= site_value * (
                    self.lattice[(x+1) % self.L, y] + self.lattice[(x-1) % self.L, y] 
                    + self.lattice[x, (y+1) % self.L] + self.lattice[x, (y-1) % self.L]
                )
        return energy / 2  # Avoid double counting
