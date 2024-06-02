import numpy as np
import sympy as sp
from sympy import Eq
import scipy
import matplotlib.pyplot as plt


# Defining symbols to be used
h, x = sp.symbols('h, x')


def G_d(x_new, x_old, delta_tau):
    """
    Importance sampling diffusive displacement probability density distribution for electrons. 
    The prefactor has been removed as we only require its relative value.
    This corresponds to Eqn 3.49 in Foulkes (2001).

        Parameters:
            x_new (float): The new x position of a walker
            x_old (float): The original x position of a walker
            delta_tau (float): The time incrament

        Returns:
            Probability density (float): The probability density of an electron moving from the specified old coordinates
                                         to the specified new coordinates
    """
    return np.exp(-((x_new - x_old - delta_tau*velocity(x_old))**2)/(2*delta_tau))


def importance_sampling_weight(old_local_energies, new_local_energies, energy_offset, delta_tau):
    """
    Time-dependent renormalization (reweighting) of the diffusion Green’s function

        Parameters:
            old[/new]_local_energies (float): The value of the local energy of the system for the old[/new] coordinates
            energy_offset (float): The value of the Diffusion Monte Carlo reference/ offset energy
            delta_tau (float): The time incrament

        Returns:
            Weight value (float): The renormalised weight of the walker associated with the given step
    """
    return np.exp(- delta_tau * (new_local_energies + old_local_energies - 2*energy_offset) / 2)


def walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps, all_positions, \
                          num_alive_walkers, max_duplicates, name, step_number):
     """
    Performs the walking and branching steps of the Diffusion Monte Carlo process.

        Parameters:
            walkers (float): A (2 x N_max) array containing the walkers initial positions and alive/dead status 
            E_T_energies (float): An empty array
            w_E_L (float): An empty array
            sum_weights (float): An initially empty array that is appended with the value of the sum of the weights
            num_steps (int): Number of steps to run the calculation for
            all_positions (float): An empty array
            num_alive_walkers (float): An empty array
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step
            name (str): 'thermalize' or 'measure' indicates if the current steps are thermalising the system or being measured
            step_number (int): A counter to keep track of the current step number

        Returns:
            walkers (float): A (2 x N_max) array containing the walkers final positions and alive/dead status 
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            all_positions (float): An array containing the position of the walkers after each measured step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
            step_number (int): Final step counter
    """

    # For each step, perform walking and branching of the walkers
    for step in range(num_steps):
        # Calculate the local energy for the given walker positions
        old_local_energies = local_energy(walkers[1, :])
        # Move the walkers by a random amount
        x_old = walkers[1, :]
        x_proposed = x_old + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + delta_tau * velocity(x_old)
        
        # Iterating over each of the walkers
        for i in range(len(walkers[1])):
            # Only if the walkers are alive
            if walkers[0, i] == 1:
                # Calculate p, the probability of accepting the steps taken by the walker. Formula given by Foulkes (2001) Eqn 3.52
                p = np.minimum(1, (G_d(x_old[i], x_proposed[i], delta_tau) * (trial_psi(x_proposed[i]))**2) \
                           / (G_d(x_proposed[i], x_old[i], delta_tau) * (trial_psi(x_old[i]))**2))
                # Metropolis acceptance criterion
                if p == 1:
                    walkers[1, i] = x_proposed[i]
                else:
                    if np.random.uniform(0, 1) <= p:
                        walkers[1, i] = x_proposed[i]
                    else:
                        walkers[1, i] = x_old[i]

        # Calculate the local energy for the new walker positions
        new_local_energies = local_energy(walkers[1, :])

        # Count the number of walkers that are alive at this step and store the number
        N_alive = sum(walkers[0, :])
        num_alive_walkers.append(N_alive)

        # Calculate the energy at this step and store the number
        E_T = np.mean(potential(walkers[1, walkers[0,:] == 1])) + alpha * (1 - N_alive / N_0)
        E_T_energies.append(E_T)

        # Calculate the weights of each walker
        weights = importance_sampling_weight(old_local_energies, new_local_energies, E_T, delta_tau)

        # Calculate the duplicity of each walker
        m_ns = np.minimum(np.floor(weights + np.random.rand(N_max)), max_duplicates + 1)

        # Find the last alive walker to add on replicated walkers after this
        N_last_alive = N_max - np.argmax((np.fliplr(walkers)[0, :]) == 1) - 1

        # Duplicate/kill walkers
        # Iterating over all the alive walkers, including the dead ones in between
        for i in range(N_last_alive):
            # If walker is alive
            if walkers[0, i] == 1:
                if m_ns[i] == 0:
                    # If duplicity is 0, kill walker
                    walkers[0, i] = 0
                elif 2 <= m_ns[i] <= int(max_duplicates + 1):
                    # Else, duplicate the walker
                    for j in range(int(m_ns[i]) - 1):
                        N_last_alive += 1
                        walkers[:, N_last_alive] = walkers[:, i]  # duplicate walker m_ns[i] times

        # For measured steps
        if str(name) == 'measure':
            # Store positions of all alive walkers
            all_positions.extend(walkers[1, walkers[0] == 1])
            # Calculate and store the value of the sum of the product of the weights and local energies, as well as the sum
            # of weights
            alive_w_E_L = []
            alive_sum_weights = 0
            for i in range(len(new_local_energies)):
                if walkers[0, i] == 1:
                    alive_w_E_L.append(new_local_energies[i])
                    alive_sum_weights += 1
            w_E_L.append(np.sum(alive_w_E_L))
            sum_weights.append(alive_sum_weights)

        step_number += 1
        print(step_number)

    return walkers, E_T_energies, w_E_L, sum_weights, all_positions, num_alive_walkers, step_number


def diffusion_monte_carlo(N_0, N_max, num_steps_thermalize, num_steps_measure, max_duplicates, delta_tau):
    """
    Perform Diffusion Monte Carlo for the 1D Harmonic Oscillator system. Sets up the initial system with guessed
    starting positions for the walkers, runs the thermalising then measuring, walking and branching steps.

        Parameters:
            N_0 (int): Initial number of alive walkers
            N_max (int): Length of array
            num_steps_thermalize_n (int): Number of steps to thermalise the system
            num_steps_measure_n (int): Number of measured steps
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step

        Returns:
            all_positions (float): An array containing the position of the walkers after each measured step
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            sum_weights (float): An array of the sum of the weights for each step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
    """
    
    # Set up matrix of walkers
    # The first dimension is indexed by 'alive/ dead status', 'x-position'
    # The second dimension is indexed by the walker number
    walkers = np.zeros((2, N_max))
    walkers[0, :N_0] = 1      # Set the first N_0 walkers to be alive
    walkers[1, :] = 0         # Starting the walkers at the origin
    num_alive_walkers = []    # List to store the number of alive walkers at each step
    all_positions = []        # List to store positions for all measurement steps
    E_T_energies = []         # List to store energy after each step
    w_E_L = []                # List to store weight*local_energy for each step
    sum_weights = []          # List to store sum of weights for each step

    # Run the simulation for the thermalising steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_thermalize, all_positions, \
                          num_alive_walkers, max_duplicates, 'thermalize', step_number)

    # Run the simulation for the measuring steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_measure, all_positions, \
                          num_alive_walkers, max_duplicates, 'measure', step_number)

    return all_positions, E_T_energies, w_E_L, sum_weights, num_alive_walkers


if __name__ == "__main__":
    N_0 = 10000                   # Initial number of alive walkers
    N_max = 10 * N_0              # Length of array
    num_steps_thermalize = 500    # Number of steps to thermalise the system
    num_steps_measure = 2500      # Number of steps for which measurements are taken
    max_duplicates = 10           # Maximum number of duplicates produced during branching step
    delta_tau = 0.01              # Time step size
    alpha = 1/delta_tau           # Degree of feedback
    n_bins = 100                  # Number of data bins
    n_wait = 250                  # Number of steps to wait before calculating the energy standard deviation

    step_number = 0               # Step number counter

    # Define the potential and trial function to be used
    potential_symbolic = 0.5*x**2
    trial_psi_symbolic = sp.exp(-(x**2)/4)

    # Corresponding velocity and local energy functions
    velocity_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, x) / trial_psi_symbolic )
    local_energy_symbolic = sp.simplify( -0.5*sp.diff(trial_psi_symbolic, x, 2) / trial_psi_symbolic ) + potential_symbolic

    # Convert symbolic expressions to functions
    potential = sp.lambdify(x, potential_symbolic)
    trial_psi = sp.lambdify(x, trial_psi_symbolic)
    velocity = sp.lambdify(x, velocity_symbolic)
    local_energy = sp.lambdify(x, local_energy_symbolic)

    # Run Diffusion Monte Carlo    
    measurements_positions, E_T_energies, w_E_L, sum_weights, num_alive_walkers \
                            = diffusion_monte_carlo(N_0, N_max, num_steps_thermalize, num_steps_measure, max_duplicates, delta_tau)

    # Calculate the exact ground state wavefunction
    h = np.linspace(-10, 10, 200000)
    psi_h = np.exp(-(h**2)/2)
    Area_int = np.trapz((psi_h)**2, h)
    x = np.linspace(-5, 5, n_bins)
    psi = np.exp(-(x**2)/2) 
    Normalised_psi = psi / np.sqrt(Area_int)

    # Plot the histogram of walker positions (wavefunction) from all measurements
    plt.figure()
    f_hist_values, f_bin_edges = np.histogram(measurements_positions, bins=n_bins, range=[-5, 5], density=True)
    wavefunction_hist_values = f_hist_values / trial_psi(x)
    wavefunction_hist_values = (wavefunction_hist_values) / np.sqrt(np.sum(wavefunction_hist_values**2 * 10/n_bins))
    plt.plot(x, wavefunction_hist_values, color='tab:purple', marker='x', ls='-', markersize = 4, label='DMC')
    plt.plot(x, Normalised_psi, label='Exact Wavefunction ', color='tab:blue')
    plt.xlabel('Position')
    plt.ylabel(r'Wavefunction, $\Psi$')
    plt.legend()
    plt.title(r'1D Harmonic Oscillator Ground State $\Psi$')
    plt.savefig('Wavefunction.png')

    # Calculate the local energy for each measured step
    E_L_energies = []
    for i in range(len(w_E_L)):
        sum_top = 0
        sum_bottom = 0
        for j in range(i+1):
            sum_top += w_E_L[j]
            sum_bottom += sum_weights[j]
        E_L_energies.append(sum_top/sum_bottom)

    # Calculate the final mean energy from all measurements
    final_mean_energy = E_L_energies[-1]
    # Calculate the standard deviation of energy from measured steps after waiting period
    running_std_energy = []
    for i in range(n_wait,len(E_L_energies)):
        running_std_energy.append(np.std(E_L_energies[:i]))
    std_energy = np.std(E_L_energies)
    mean_std_energy = std_energy
    print("Final Mean E_L from all Measurements:", final_mean_energy)
    print("Standard Error of E_L from Measured Steps:", std_energy)
    print("Exact Energy: 0.5")

    # Plot the energy variation with step number
    plt.figure()
    plt.plot(np.arange(len(E_L_energies)), E_L_energies, color='tab:green', label='DMC Energy')
    upper_limit = np.array(E_L_energies[n_wait:]) + np.array(running_std_energy)
    lower_limit = np.array(E_L_energies[n_wait:]) - np.array(running_std_energy)
    plt.plot(np.arange(n_wait,len(E_L_energies)), upper_limit, color='tab:olive', label='Standard Error')
    plt.plot(np.arange(n_wait,len(E_L_energies)), lower_limit, color='tab:olive')
    plt.fill_between(np.arange(n_wait,len(E_L_energies)), lower_limit, upper_limit, color='tab:olive', alpha=0.5)
    plt.xlabel('Measured Time Step')
    plt.ylabel(r'Energy / $\hbar \omega$')
    plt.title(r'Variation of E$_L$ with Step Number')
    plt.axhline(y=0.5, color='tab:red', linestyle='dashed', label='Exact Energy')
    plt.legend()
    plt.savefig('Energy.png')

    # Plot the number of alive walkers against step number
    plt.figure()
    plt.plot(np.arange(len(num_alive_walkers)), num_alive_walkers, color='tab:cyan')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Alive Walkers')
    plt.title('Variation of Number of Alive Walkers with Step Number')
    plt.savefig('Alive_walkers.png')
