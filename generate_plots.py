import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.integrate import trapz

def read_ising_data(L):
    """
    Read in thermodynamic data for a lattice simulation with dimension L.
    
    Returns a set of five 1D arrays, corresponding to temperature, total energy,
    specific heat, magnetization, and magnetic susceptibility, respectively.
    """
    path = "data/thermodynamics/L" + str(L) + ".csv"
    df = pd.read_csv(path)
    T_data = df["temperature"].to_numpy()
    E_data = df["total_energy"].to_numpy()
    C_data = df["specific_heat"].to_numpy()
    M_data = df["magnetization_per_site"].to_numpy()
    X_data = df["magnetic_susceptibility"].to_numpy()
    return T_data, E_data, C_data, M_data, X_data

def TC_at_L_scaling(L, TC, x_0, nu):
    """
    Function describing the finite-size scaling effects on the critical temperature.
    """
    return TC + (x_0*TC)*(L**(-1/nu))

def get_peak_coordinates(x, y):
    """
    Returns the (x, y) coordinates where y is maximized.
    """
    amax = np.argmax(y)
    x_at_peak = x[amax]
    y_peak = y[amax]
    return x_at_peak, y_peak

def linear_fit(x, y):
    """
    Performs a linear regression of y on x.
    """
    linreg = linregress(x, y)
    m = linreg[0]
    b = linreg[1]
    return m, b

def E_vs_T(dims, T_arrays, E_arrays):
    """
    Plots the average energy per lattice site as a function of temperature, for all
    lattice dimensions.
    """
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, E_arr in zip(dims, T_arrays, E_arrays):
        ax.plot(T_arr, E_arr/L**2, label = "L = " + str(L), markersize=8)

    ax.set_xlabel("$T \ [J\ /\ k_B]$", size = 12)
    ax.set_ylabel("$E/N \ [J]$", size = 12)
    ax.legend(fontsize = 12)
    ax.grid()
    fig.savefig("plots/E_vs_T.png")

def X_vs_T(dims, T_arrays, X_arrays):
    """
    Plots the magnetic susceptibility as a function of temperature, for all
    lattice dimensions.
    """
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, X_arr in zip(dims, T_arrays, X_arrays):
        ax.plot(T_arr, X_arr, label = "L = " + str(L), markersize=8)

    ax.set_xlabel("$T \ [J\ /\ k_B]$", size = 12)
    ax.set_ylabel("$\chi$", size = 12)
    ax.legend(fontsize = 12)
    ax.grid()
    fig.savefig("plots/X_vs_T.png")

def T_CL_vs_Linv_X(dims, T_arrays, X_arrays, nu=1):
    """
    Plots the critical temperature as determined for each lattice size as a function 
    of the inverse of the dimension, L**(-1). Each "dimension-dependent" critical
    temperature is defined as the temperature at which the magnetic susceptibility peaks.
    A fit is performed to extract the true thermodynamic critical temperature.

    Assumes an exact critical exponent nu = 1.
    """
    # Extract the critical temperatures in each lattice simulation, i.e. the temperatures
    # at which the magnetic susceptibility peaks
    TCL_X = []
    for T_arr, X_arr in zip(T_arrays, X_arrays):
        T_at_X_peak, X_peak = get_peak_coordinates(T_arr, X_arr)
        TCL_X.append(T_at_X_peak)
    TCL_X = np.array(TCL_X)

    # Perform a fit to extract the thermodynamic critical temperature, ideally
    # unaffected by finite-size effects.
    popt, pcov = curve_fit(lambda L, TC, x_0: TC_at_L_scaling(L, TC, x_0, 1), dims, TCL_X)
    
    TC_X = popt[0] # T_C
    m = popt[1] * popt[0] # (x_0 * T_C)
    x = np.linspace(dims[-1]**(-1/nu), dims[0]**(-1/nu), 1000)
    y = TC_X + m*x

    L_invs = np.array([L**(-1/nu) for L in dims])

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))
    for L, L_inv, T in zip(dims, L_invs, TCL_X):
        ax.scatter(L_inv, T, label = "L=" + str(L))

    ax.grid()
    ax.plot(x, y, linestyle = '--', c='green', label = r"Fit: $T_C(L)=T_C + (x_0T_c)L^{-1/\nu}$")
    ax.set_xlabel("$L^{-1}$", size=12)
    ax.set_ylabel("$T_C(L) \ [J\ /\ k_B]$", size=12)
    ax.legend(fontsize=12)
    print("T_C ~ " + str(round(popt[0], 3)))
    fig.savefig("plots/T_CL_vs_Linv_X.png")

def log_log_X_vs_L(dims, T_arrays, X_arrays):
    """
    Plot the peak magnetic susceptibility for each lattice dimension as a function of the lattice
    dimension, but on a log-log scale. This plot should demonstrate a linear relationship.

    Returns the critical exponent beta/nu derived from a linear fit to the data. The exact value is
    beta/nu = 1.75.
    """
    X_at_TC = []
    for T_arr, X_arr in zip(T_arrays, X_arrays):
        T_at_X_peak, X_peak = get_peak_coordinates(T_arr, X_arr)
        X_at_TC.append(X_peak)
        
    logX_at_TC = np.log(np.array(X_at_TC))
    log_L_list = np.log(dims)
    gamma_nu_fit, yint = linear_fit(log_L_list, logX_at_TC)

    x = np.linspace(np.log(dims[0]), np.log(dims[-1]), 1000)
    y = yint + gamma_nu_fit*x

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, log_L, logX in zip(dims, log_L_list, logX_at_TC):
        ax.scatter(log_L, logX, label = "L=" + str(L), s=20)

    ax.plot(x,y,linestyle='--',color='black', alpha=0.5, label="Linear fit")
    ax.set_xlabel("ln$(L)$", fontsize=12)
    ax.set_ylabel("ln$(\chi [T_C(L)])$", fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)

    # Uncomment following line to print text on plot. The first two arguments
    # (text coordinates) may need to be adjusted.
    #ax.text(3.09, 1.6, r"$\gamma / \nu \approx $" + str(round(gamma_nu_fit, 3)), size = 12)

    print("gamma/nu ~ " + str(round(gamma_nu_fit, 3)))

    fig.savefig("plots/log_log_X_vs_L.png")

    return gamma_nu_fit

def scaled_X_vs_scaled_T(dims, T_arrays, X_arrays, gamma_nu_fit, nu=1):
    """
    Plots the magnetic susceptibility as a function of temperature, but with both
    variables scaled to account for finite-size effects. The susceptibility curves
    should overlap around the critical temperature.
    """
    TCs = []
    for T_arr, X_arr in zip(T_arrays, X_arrays):
        T_at_X_peak, X_peak = get_peak_coordinates(T_arr, X_arr)
        TCs.append(T_at_X_peak)

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, X_arr, TC_X in zip(dims, T_arrays, X_arrays, TCs):
        X_scaled = L**(-gamma_nu_fit) * X_arr
        T_scaled = L**(1/nu) * (T_arr - TC_X)
        ax.plot(T_scaled, X_scaled, markersize=8, label="L=" + str(L))

    ax.set_xlabel(r"$L^{1/\nu}(T-T_c(L)) \ [J\ / \ k_B]$", fontsize=12)
    ax.set_ylabel(r"$L^{-\gamma/\nu}\chi$", fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)

    # Uncomment following lines to print text on plot. The first two arguments in each line
    # (text coordinates) may need to be adjusted.
    #ax.text(-65, .025, r"$\nu = $" + str(round(nu, 3)), size = 12)
    #ax.text(-65, .023, r"$\gamma/\nu \approx $" + str(round(gamma_nu_fit, 3)), size = 12)

    fig.savefig("plots/scaled_X_vs_scaled_T.png")

def M_vs_T(dims, T_arrays, M_arrays):
    """
    Plots the absolute value of the average magnetization per site as a function of
    temperature, for all lattice dimensions.
    """
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, M_arr in zip(dims, T_arrays, M_arrays):
        ax.plot(T_arr, [abs(M) for M in M_arr], markersize=8, label = "L = " + str(L))

    ax.set_xlabel("$T \ [J\ / \ k_B]$", fontsize=12)
    ax.set_ylabel(r"$M$", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()

    fig.savefig("plots/M_vs_T.png")

def M_intersect(dims, T_arrays, M_arrays, beta_nu=0.25):
    """
    Plots the absolute magnetization, scaled by the lattice dimension to a critical
    exponent beta/nu, as a function of temperature. 

    Returns the critical exponent beta_nu as well as the extracted critical temperature
    TC_M.
    
    By default, beta_nu is set to 0.25 (the exact value in the 2D Ising model solution).
    For the purposes of this analysis, the user should adjust beta/nu to provide the 
    best intersection of the four magnetization curves (visible in the inset). The user
    should also adjust the "offset" variable to further determine the best critical
    temperature for which all four curves intersect.
    
    If the user wishes not to perform this manual step, the user may simply use the default
    beta/nu = 0.25, which should still demonstrate the correct critical behavior.
    """
    # Adjust beta_nu as needed
    beta_nu = 0.13

    M_scaled_arrays = []
    for L, M_arr in zip(dims, M_arrays):
        M_scaled_arrays.append(L**beta_nu * M_arr)
    M_scaled_arrays = np.array(M_scaled_arrays)

    # Finds the intersection temperature (between the curves corresponding to smallest and 
    # largest lattice sizes, which is an arbitrary choice)
    T_int = np.argwhere(np.diff(np.sign(M_scaled_arrays[0] - M_scaled_arrays[-1]))).flatten()

    # Adjust offset as needed
    offset = 0.0065

    TC_M = (T_arrays[0])[T_int][0] + offset

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(7,5))
    
    ins = ax.inset_axes([0.1, 0.2, 0.25, 0.4])
    ins.set_yticklabels([])
    ins.tick_params(left=False)
    lb = 148
    ub = 155

    for L, T_arr, M_scaled_arr in zip(dims, T_arrays, M_scaled_arrays):
        ax.plot(T_arr, [abs(M) for M in M_scaled_arr], markersize=8, label = "L = " + str(L))
        ins.plot(T_arr[lb:ub], [abs(M) for M in M_scaled_arr[lb:ub]])

    ax.axvline(TC_M, linestyle='--', color='grey', alpha=0.5, label="$T_C$ (intersection)")
    ax.set_xlabel(r"$T \ [J\ / \ k_B]$", fontsize=12)
    ax.set_ylabel(r"$L^{\beta/\nu}M$", fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)
    ins.axvline(TC_M, linestyle='--', color='grey', alpha=0.5)

    # Uncomment following lines to print text on plot. The first two arguments in each line
    # (text coordinates) may need to be adjusted.
    #ax.text(3.1, 0.92, r"$T_C \approx $" + "%.3f" % round(TC_M, 3), fontsize=12)
    #ax.text(3.1, 0.85, r"$\beta/\nu \approx $" + "%.2f" % round(beta_nu, 3), fontsize=12)

    print("T_C ~ " + str(round(TC_M, 3)))
    print("beta/nu ~ " + str(round(beta_nu, 3)))

    fig.savefig("plots/M_intersect.png")

    return beta_nu, TC_M

def scaled_M_vs_scaled_T(dims, T_arrays, M_arrays, beta_nu, TC_M, nu=1):
    """
    Plots the absolute magnetization as a function of temperature, but with both
    variables scaled to account for finite-size effects. The magnetization curves
    should overlap around the critical temperature.
    """
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, M_arr in zip(dims, T_arrays, M_arrays):
        M_scaled = L**(beta_nu) * np.array([abs(M) for M in M_arr])
        T_scaled = L**(1/nu) * (T_arr - TC_M)
        ax.plot(T_scaled, M_scaled, markersize=8, label="L=" + str(L))

    ax.set_xlabel(r"$L^{1/\nu}(T-T_c(L)) \ [J\ /\ k_B]$", fontsize=12)
    ax.set_ylabel(r"$L^{\beta/\nu}M$", fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)

    fig.savefig("plots/scaled_M_vs_scaled_T.png")

def C_vs_T(dims, T_arrays, C_arrays):
    """
    Plots the specific heat as a function of temperature for all lattice dimensions.
    """
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, C_arr in zip(dims, T_arrays, C_arrays):
        ax.plot(T_arr, C_arr, markersize=8, label="L=" + str(L))

    ax.set_xlabel("$T \ [J\ /\ k_B]$", fontsize=12)
    ax.set_ylabel("$C_V \ [k_B]$", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()

    fig.savefig("plots/C_vs_T.png")

def T_CL_vs_Linv_C(dims, T_arrays, C_arrays, nu=1):
    """
    Plots the critical temperature as determined for each lattice size as a function 
    of the inverse of the dimension, L**(-1). Each "dimension-dependent" critical
    temperature is defined as the temperature at which the specific heat peaks. A fit 
    is performed to extract the true thermodynamic critical temperature.

    Assumes an exact critical exponent nu = 1.
    """
    TCL_C = []
    for T_arr, C_arr in zip(T_arrays, C_arrays):
        T_at_C_peak, C_peak = get_peak_coordinates(T_arr, C_arr)
        TCL_C.append(T_at_C_peak)

    TCL_C = np.array(TCL_C)

    popt0, pcov0 = curve_fit(lambda L, TC, x_0: TC_at_L_scaling(L, TC, x_0, 1), dims, TCL_C)

    TC_C = popt0[0] # T_C
    m = popt0[1] * popt0[0] # (x_0 * T_C)
    x = np.linspace(dims[-1]**(-1/nu), dims[0]**(-1/nu), 1000)
    y = TC_C + m*x

    L_invs = np.array([L**(-1/nu) for L in dims])

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, L_inv, T in zip(dims, L_invs, TCL_C):
        ax.scatter(L_inv, T, label = "L=" + str(L))

    ax.grid()
    ax.plot(x, y, linestyle = '--', c='grey', label = r"Fit: $T_C(L)=T_C + (x_0T_c)L^{-1/\nu}$")
    ax.set_xlabel("$L^{-1}$", size=12)
    ax.set_ylabel("$T_C(L) \ [J\ /\ k_B]$", size=12)
    ax.legend(fontsize=12)
    print("T_C ~ " + str(round(popt0[0], 3)))

    fig.savefig("plots/T_CL_vs_Linv_C.png")

def C_max_vs_log_L(dims, T_arrays, C_arrays):
    """
    Plots the peak specific heat as a function of the log of the lattice dimension.
    The data should demonstrate a linear relationship.
    """
    C_maxs = []
    for T_arr, C_arr in zip(T_arrays, C_arrays):
        T_at_C_peak, C_peak = get_peak_coordinates(T_arr, C_arr)
        C_maxs.append(C_peak)
    C_maxs = np.array(C_maxs)
    log_L_list = np.log(dims)

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, log_L, C_max in zip(dims, log_L_list, C_maxs):
        ax.scatter(log_L, C_max, s=20, label="L=" + str(L))

    ax.set_xlabel(r"ln$(L)$", fontsize=12)
    ax.set_ylabel(r"$C_{max}(L) \ [k_B]$", fontsize=12)
    ax.grid()
    ax.legend(fontsize=12)

    fig.savefig("plots/C_max_vs_log_L")

def S_vs_T(dims, T_arrays, C_arrays):
    """
    Plots the entropy of the configuration as a function of temperature, for all lattice
    dimensions.

    The entropy S(T) is determined by integrating C(T')/T' from T'=0 to T'=T.

    Returns a numpy array of 1D arrays, each of which corresponds to the entropy values
    for a given lattice dimension.
    """

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))
    
    S_arrays = []
    for L, T_arr, C_arr in zip(dims, T_arrays, C_arrays):
        C_by_T = C_arr / T_arr
        S_arr = []

        for i in range(len(T_arr)):
            Ts = T_arr[:i]
            C_by_Ts = C_by_T[:i]
            S_arr.append(trapz(C_by_Ts, Ts))

        S_arr = np.array(S_arr)
        S_arrays.append(S_arr)
        ax.plot(T_arr, S_arr, markersize=8, label="L=" + str(L))

    ax.axhline(np.log(2), linestyle = "--", color = "grey", label = "ln(2)")
    ax.set_xlabel("$T \ [J\ /\k_B]$", fontsize=12)
    ax.set_ylabel("$S \ [k_B]$", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()
    
    fig.savefig("plots/S_vs_T")

    return np.array(S_arrays)

def F_vs_T(dims, T_arrays, E_arrays, S_arrays):
    """
    Plots the free energy per lattice site F=E/N-TS as a function of temperature.
    (The 1/N factor multiplying E is specific to this analysis, since we define E as extensive
    but define S to already be intensive).
    """
    
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,5))

    for L, T_arr, E_arr, S_arr in zip(dims, T_arrays, E_arrays, S_arrays):
        F_arr = E_arr/L**2 - T_arr * S_arr
        ax.plot(T_arr, F_arr, markersize=8, label="L=" + str(L))

    ax.set_xlabel("$T \ [J\ /\ k_B]$", fontsize=12)
    ax.set_ylabel("$F_N = E/N - TS \ [J]$", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()

    fig.savefig("plots/F_vs_T")

def plot_lattices(lattices):
    """
    Plots three lattice spin configurations at different temperatures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6, 5)) 

    i=0
    for ax, lattice in zip(axes, lattices):
        color_map = (lattice + 1) / 2  # Scale -1 to 0 and +1 to 1
        ax.imshow(color_map, cmap="gray", vmin=0, vmax=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False,bottom=False)

        # Change labels if desired.
        if i==0:
            ax.set_xlabel(r"$T < T_C$", fontsize = 12)
        elif i==1:
            ax.set_xlabel(r"$T \approx T_C$", fontsize = 12)
        else:
            ax.set_xlabel(r"$T > T_C$", fontsize = 12)
        i += 1

    plt.tight_layout()

    fig.savefig("plots/lattices.png")


def main():

    ##### Plotting parameters #####

    # Array of lattice dimensions
    dims = [10, 16, 24, 36]
    dims = sorted(dims)

    # Index corresponding to lattice for plotting spin configurations (see plot_lattices())
    # By default, the largest lattice is plotted
    idx_to_plot = -1

    if not os.path.exists("plots"):
        os.makedirs("plots")

    T_arrays = []; E_arrays = []; C_arrays = []; M_arrays = []; X_arrays = []
    for L in dims:
        # Read in thermodynamic data for each lattice simulated
        T_array, E_array, C_array, M_array, X_array = read_ising_data(L)
        T_arrays.append(T_array)
        E_arrays.append(E_array)
        C_arrays.append(C_array)
        M_arrays.append(M_array)
        X_arrays.append(X_array)

    # Energy plots
    E_vs_T(dims, T_arrays, E_arrays)

    # Magnetic susceptibility plots
    print()
    print("Magnetic susceptibility analysis")
    print("________________________________")
    X_vs_T(dims, T_arrays, X_arrays)
    T_CL_vs_Linv_X(dims, T_arrays, X_arrays, nu=1)
    gamma_nu_fit = log_log_X_vs_L(dims, T_arrays, X_arrays)
    scaled_X_vs_scaled_T(dims, T_arrays, X_arrays, gamma_nu_fit, nu=1)

    # Magnetization plots
    print()
    print("Magnetism analysis")
    print("__________________")
    M_vs_T(dims, T_arrays, M_arrays)
    beta_nu, TC_M = M_intersect(dims, T_arrays, M_arrays)
    scaled_M_vs_scaled_T(dims, T_arrays, M_arrays, beta_nu, TC_M, nu=1)
    
    # Specific heat plots
    print()
    print("Specific heat analysis")
    print("______________________")
    C_vs_T(dims, T_arrays, C_arrays)
    T_CL_vs_Linv_C(dims, T_arrays, C_arrays, nu=1)
    C_max_vs_log_L(dims, T_arrays, C_arrays)

    # Entropy, free energy plots
    S_arrays = S_vs_T(dims, T_arrays, C_arrays)
    F_vs_T(dims, T_arrays, E_arrays, S_arrays)

    print()

    # Lattice plotting
    lattice_arrays = []

    # By default, largest lattice size chosen for plotting. User may change if desired.
    lattice_dir = "data/lattices/L" + str(dims[idx_to_plot])
    for filename in sorted(os.listdir(lattice_dir)):
        f = os.path.join(lattice_dir, filename)
        lattice_arrays.append(pd.read_csv(f, header=None).to_numpy())

    plot_lattices(lattice_arrays)

if __name__ == "__main__":
    main()
