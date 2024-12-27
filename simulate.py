import numpy as np
from metropolis import step
from lattice import Lattice
import sys
import csv
from tqdm import tqdm
import os


def simulate(L, T, n_thermalize, n_analyze, n_measure_interval, save_lattice):
    """ 
    Simulates the square 2D Ising Model with specified input dimensions and temperature

    Parameters 
    ----------
    L: int
        Lattice dimension
    T: float
        Lattice temperature
    n_thermalize: int
        Number of thermalization sweeps to conduct
    n_analyze: int
        Number of analysis sweeps to conduct
    n_measure_interval: int
        Number of sweeps between each measurement of thermodynamic quantities (n_measure_interval < n_analyze)
    save_lattice: bool
        Indicates whether the spin layout of the lattice should be saved to a csv file

    Returns
    -------
    total_energy: float
        Total energy of the lattice, averaged over all measurements at the given temperature
    total_squared_energy: float
        Total squared energy of the lattice, averaged over all measurements at the given temperature
    mag: float
        Average (absolute) magnetization per site, additionally averaged over all measurements at the given temperature
    squared_mag:
        Average squared magnetization per site, additionally averaged over all measurements at the given temperature
    """

    # Counter to keep track of number of measurements made
    n_measure_made = 0 
    
    # Initialize the lattice
    lattice = Lattice(L, T)

    # Print lattice dimension and temperature
    print('\n')
    print("L = " + str(L) + ", T = " + str(round(T,3)) + '\n')

    # Perform thermalization sweeps
    for i in tqdm(range(n_thermalize), total=n_thermalize, desc="Thermalization sweeps"):
        step(L, lattice, T)

    # Initialize variables to store thermodynamic quantities across simulation steps
    total_energy = 0
    total_squared_energy = 0
    mag = 0
    squared_mag = 0
    
    # Perform analysis sweeps (with progress updates)
    for i in tqdm(range(n_analyze), total=n_analyze, desc="Analysis sweeps"):

        step(L, lattice, T)

        # Every n_measure_interval sweeps, record a measurement
        if (i % n_measure_interval == 0):

            # Get lattice energy
            cur_config_energy = lattice.get_total_energy()

            # Add energy and squared energy to respective variables
            total_energy += cur_config_energy
            total_squared_energy += (cur_config_energy**2) # Take ensemble average to get <H^2>

            # Get (absolute) lattice magnetization per site
            cur_mag = lattice.get_mag()

            # Add magnetization and squared magnetization to respective variables
            mag += cur_mag
            squared_mag += (cur_mag**2)

            # Update measurement counter
            n_measure_made += 1

    # Average thermodynamic quantities over all measurements
    total_energy /= n_measure_made
    total_squared_energy /= n_measure_made
    mag /= n_measure_made
    squared_mag /= n_measure_made

    # Save lattice as a csv file (if applicable)
    if save_lattice:
        T_rounded = ("%.3f" % round(T, 3)).replace(".", "p")
        lattice_csv_path = "data/lattices/L" + str(L) + "/T" + T_rounded + ".csv"
        with open(lattice_csv_path, "w+") as f:
            np.savetxt(f, lattice.lattice, delimiter=",")

    return total_energy, total_squared_energy, mag, squared_mag


def write_data(L, T_list, E_list, C_list, M_list, chi_list):
    """ 
    Write thermodynamic data from the simulation over a range of temperatures to a csv file 

    Parameters 
    ----------
    L: int
        Lattice dimension
    T_list: numpy 1D array of floats
        Array of simulated temperatures
    E_list: numpy 1D array of floats
        Array of total energies, each averaged over all measurements at given temperature 
    C_list: numpy 1D array of floats
        Array of specific heats, each averaged over all measurements at given temperature 
    M_list: numpy 1D array of floats
        Array of average (absolute) site magnetization, each averaged over all measurements at given temperature
    chi_list: numpy 1D array of floats
        Array of magnetic susceptibilities, each averaged over all measurements at given temperature 
    save_lattice: bool
        Indicates whether the spin layout of the lattice should be saved to a csv file

    Returns: void
        write_data() writes to a csv the total energy, specific heat, absolute magnetization per site, and magnetic susceptibility
        at each temperature after averaging over all measurements. The csv is written to data/lattices.
    """

    # Output path for spin configuration csv
    path = "data/thermodynamics/L" + str(L) + ".csv"

    # Create rows for csv
    rows = zip(T_list, E_list, C_list, M_list, chi_list)

    with open(path, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "total_energy", "specific_heat", "magnetization_per_site", "magnetic_susceptibility"])
        for row in rows:
            writer.writerow(row)

    return

def main():

    ##### Simulation parameters #####

    # Array of lattice dimensions to simulate
    dims = [10, 16, 24, 36]

    # Lower and upper bound on temperatures to simulate
    T_low = 0.015
    T_high = 4.5
    if T_high < T_low:
        sys.exit("Invalid temperature range.")

    # Temperature step size
    T_step = 0.015
    if T_step > (T_high - T_low):
        sys.exit("Temperature step size exceeds temperature range.")
    
    # Number of thermalization sweeps
    n_thermalize = 100000

    # Number of analysis sweeps
    n_analyze = 300000

    # Sweep interval at which measurements are taken (over the course of the analysis sweeps)
    n_measure_interval = 10
    if n_measure_interval > n_analyze:
        sys.exit("Measurement interval greater than number of analysis sweeps.")

    # Temperature percentiles at which to save lattice layout (must select three)
    T_percentiles = [0.1, 0.5, 0.9]
    if len(T_percentiles) != 3:
        sys.exit("Must select three percentiles of temperature input array at which spin layout of lattice is saved.")
    for T in T_percentiles:
        if (T < 0.0) or (T > 1.0):
            sys.exit("One or more temperature percentiles are invalid (below zero or above one).")

    #################################

    # Make necessary directories if they do not exist
    if not os.path.exists("data/thermodynamics"):
        os.makedirs("data/thermodynamics")
    if not os.path.exists("data/lattices"):
        os.makedirs("data/lattices")
    for L in dims:
        if not os.path.exists("data/lattices/L" + str(L)):
            os.makedirs("data/lattices/L" + str(L))

    # Initialize an array of temperatures to simulate
    T_in = np.arange(T_low, T_high, T_step)

    T_low_p = T_percentiles[0]; T_low_save_idx = round((len(T_in) - 1) * T_low_p)
    T_mid_p =  T_percentiles[1]; T_mid_save_idx = round((len(T_in) - 1) * T_mid_p)
    T_high_p = T_percentiles[2]; T_high_save_idx = round((len(T_in) - 1) * T_high_p)

    # Initialize arrays to store thermodynamic quantities
    E_list = np.zeros(len(T_in), dtype=float)
    C_list = np.zeros(len(T_in), dtype=float)
    M_list = np.zeros(len(T_in), dtype=float)
    chi_list = np.zeros(len(T_in), dtype=float)

    for L in dims:

        for t in range(len(T_in)):

            T = T_in[t]

            # Decide whether or not to save the lattice at the given temperature
            save_lattice = False
            if (t == T_low_save_idx) or (t == T_mid_save_idx) or (t == T_high_save_idx):
                save_lattice = True

            # Simulate
            total_energy, total_squared_energy, mag, squared_mag = simulate(L, T, n_thermalize, n_analyze, n_measure_interval, save_lattice)

            # Energy
            E_list[t] = total_energy

            # Specific heat
            C = (1/((L**2)*(T**2)))*((total_squared_energy - total_energy**2))
            C_list[t] = C

            # Magnetization per site
            M_list[t] = mag

            # Magnetic susceptibility
            chi = ((L**2)/T)*(squared_mag - mag**2)
            chi_list[t] = chi

        write_data(L, T_in, E_list, C_list, M_list, chi_list)
    
if __name__ == "__main__":
    main()