import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as plt


# Defining symbols to be used
h, x, y, z, r = sp.symbols('h, x, y, z, r')


def G_d(x_new, x_old, y_new, y_old, z_new, z_old, delta_tau):
    """
    Importance sampling diffusive displacement probability density distribution for the electron. 
    The prefactor has been removed as we only require its relative value.
    This corresponds to Eqn 3.49 in Foulkes (2001).
    The value corresponds to the product of the probability densities for the independent x, y and z displacements.

        Parameters:
            x[/y/z]_old (float): The original x[/y/z] position of an electron
            x[/y/z]_new (float): The new x[/y/z] position of an electron
            delta_tau (float): The time incrament

        Returns:
            Probability density (float): The probability density of an electron moving from the specified old coordinates
                                         to the specified new coordinates
    """
    return np.exp(-((x_new - x_old - delta_tau * velocity_x(x_old,y_old,z_old))**2)/(2*delta_tau)) * \
           np.exp(-((y_new - y_old - delta_tau * velocity_y(x_old,y_old,z_old))**2)/(2*delta_tau)) * \
           np.exp(-((z_new - z_old - delta_tau * velocity_z(x_old,y_old,z_old))**2)/(2*delta_tau))


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


def walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps, \
                          all_x_positions, all_y_positions, all_z_positions, num_alive_walkers, \
                          max_duplicates, name, step_number):
    """
    Performs the walking and branching steps of the Diffusion Monte Carlo process.

        Parameters:
            walkers (float): A (3 x N_max) array containing the walkers initial positions and alive/dead status 
            E_T_energies (float): An empty array
            w_E_L (float): An empty array
            sum_weights (float): An initially empty array that is appended with the value of the sum of the weights
            num_steps (int): Number of steps to run the calculation for
            all_x[/y/z]_positions (float): An empty array
            num_alive_walkers (float): An empty array
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step
            name (str): 'thermalize' or 'measure' indicates if the current steps are thermalising the system or being measured
            step_number (int): A counter to keep track of the current step number
            delta_tau (float): The time increment
            alpha (float): The degree of feedback

        Returns:
            walkers (float): A (3 x N_max) array containing the walkers final positions and alive/dead status 
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            all_x[/y/z]_positions (float): An array containing the x[/y/z] position of the electron after each measured step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
            step_number (int): Final step counter
    """

    # For each step, perform walking and branching of the walkers
    for step in range(num_steps):
        # Calculate the local energy for the given walker positions
        old_local_energies = local_energy(walkers[1, walkers[0,:]==1], \
                                          walkers[2, walkers[0,:]==1], \
                                          walkers[3, walkers[0,:]==1])

        # Move the walkers by a random amount
        x_old = walkers[1, :]
        y_old = walkers[2, :]
        z_old = walkers[3, :]
        x_proposed = x_old + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + delta_tau * velocity_x(x_old,y_old,z_old)
        y_proposed = y_old + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + delta_tau * velocity_y(x_old,y_old,z_old)
        z_proposed = z_old + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + delta_tau * velocity_z(x_old,y_old,z_old)

        # Iterating over each of the walkers
        for i in range(len(walkers[1])):
            # Only if the walkers are alive
            if walkers[0, i] == 1:
                # Calculate p, the probability of accepting the steps taken by the walker. Formula given by Foulkes (2001) Eqn 3.52
                p = np.minimum(1, (G_d(x_old[i], x_proposed[i], y_old[i], y_proposed[i], z_old[i], z_proposed[i], delta_tau) * (trial_psi(x_proposed[i], y_proposed[i], z_proposed[i]))**2) \
                           / (G_d(x_proposed[i], x_old[i], y_proposed[i], y_old[i], z_proposed[i], z_old[i], delta_tau) * (trial_psi(x_old[i], y_old[i], z_old[i]))**2))
                # Metropolis acceptance criterion
                if p == 1:
                    walkers[1, i] = x_proposed[i]
                    walkers[2, i] = y_proposed[i]
                    walkers[3, i] = z_proposed[i]
                else:
                    if np.random.uniform(0, 1) <= p:
                        walkers[1, i] = x_proposed[i]
                        walkers[2, i] = y_proposed[i]
                        walkers[3, i] = z_proposed[i]
                    else:
                        walkers[1, i] = x_old[i]
                        walkers[2, i] = y_old[i]
                        walkers[3, i] = z_old[i]

        # Calculate the local energy for the new walker positions
        new_local_energies = local_energy(walkers[1, walkers[0,:]==1], \
                                          walkers[2, walkers[0,:]==1], \
                                          walkers[3, walkers[0,:]==1])

        # Count the number of walkers that are alive at this step and store the number
        N_alive = sum(walkers[0, :])
        num_alive_walkers.append(N_alive)

        # Calculate the reference/ offset energy at this step and store the number
        E_T = np.mean(potential(walkers[1, walkers[0,:] == 1], \
                                walkers[2, walkers[0,:] == 1], \
                                walkers[3, walkers[0,:] == 1])) \
                                + alpha * (1 - N_alive / N_0)
        E_T_energies.append(E_T)

        # Calculate the weights of each walker
        weights = importance_sampling_weight(old_local_energies, new_local_energies, E_T, delta_tau)

        # Calculate the duplicity of each walker
        m_ns = np.minimum(np.floor(weights + np.random.rand(len(walkers[0, walkers[0,:] == 1]))), max_duplicates + 1)

        # Find the last alive walker to add on replicated walkers after this
        N_last_alive = N_max - np.argmax((np.fliplr(walkers)[0, :]) == 1) - 1

        # If the array is nearly filled, make a new walkers array only containing current alive walkers and extra space to fill
        if N_last_alive >= 0.95 * N_max:
            number_of_alive = int(sum(walkers[0, :]))
            walkers[:, :number_of_alive] = walkers[:, walkers[0,:] == 1]
            walkers[0, :number_of_alive] = 1
            walkers[0, number_of_alive:] = 0
            #print('RESET')

        # Find the last alive walker of the new array to add on replicated walkers after this
        N_last_alive = N_max - np.argmax((np.fliplr(walkers)[0, :]) == 1) - 1

        # Duplicate/ kill walkers
        # Define a counter to keep track of the alive walkers
        counter = -1
        # Iterating over all the alive walkers, including the dead ones in between
        for i in range(N_last_alive):
            # If walker is alive
            if walkers[0, i] == 1:
                # Increase counter by 1 to correctly iterate through the weights array
                counter += 1
                if m_ns[counter] == 0:
                    # If duplicity is 0, kill walker
                    walkers[0, i] = 0
                elif 2 <= m_ns[counter] <= int(max_duplicates + 1):
                    # Else, duplicate the walker
                    for j in range(int(m_ns[counter]) - 1):
                        N_last_alive += 1
                        walkers[:, N_last_alive] = walkers[:, i]

        # For measured steps
        if str(name) == 'measure':
            # Store positions of all alive walkers
            all_x_positions.extend(walkers[1, walkers[0] == 1])
            all_y_positions.extend(walkers[2, walkers[0] == 1])
            all_z_positions.extend(walkers[3, walkers[0] == 1])
            # Calculate and store the value of the sum of the product of the weights and local energies, as well as the sum
            # of weights
            alive_w_E_L = []
            alive_sum_weights = []
            for i in range(len(new_local_energies)):
                if walkers[0, i] == 1:
                    alive_w_E_L.append(weights[i] * new_local_energies[i])
                    alive_sum_weights.append(weights[i])
            w_E_L.append(np.sum(alive_w_E_L))
            sum_weights.append(np.sum(alive_sum_weights))

        # Increase step counter by 1
        step_number += 1
        #print(step_number)

    return walkers, E_T_energies, w_E_L, sum_weights, all_x_positions, all_y_positions, all_z_positions, \
           num_alive_walkers, step_number


def diffusion_monte_carlo(N_0, N_max, num_steps_thermalize, num_steps_measure, max_duplicates, delta_tau):
    """
    Perform Diffusion Monte Carlo for the Hydrogen atom. Sets up the initial system with guessed starting positions for the walkers,
    runs the thermalising then measuring, walking and branching steps.

        Parameters:
            N_0 (int): Initial number of alive walkers
            N_max (int): Length of array
            num_steps_thermalize_n (int): Number of steps to thermalise the system
            num_steps_measure_n (int): Number of measured steps
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step
            delta_tau (float): The time increment       

        Returns:
            all_x[/y/z]_positions (float): An array containing the x[/y/z] position of the electron after each measured step
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            sum_weights (float): An array of the sum of the weights for each step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
    """

    # The first dimension is indexed by 'alive/ dead status', 'x-position', 'y-position', 'z-position'
    # The second dimension is indexed by the walker number
    walkers = np.zeros((4, N_max))                          # Set the first N_0 walkers to be alive

    walkers[0, :N_0] = 1                                    # Set the first N_0 walkers to be alive
    walkers[1:, :] =  np.random.uniform(-2, 2, [3,N_max])   # Starting the walkers somewhat spread out from origin
    E_T_energies = []                                       # List to store energy after each step
    w_E_L = []                                              # Array to store sum of product of weight and local energy after each measured step
    sum_weights = []                                        # Array to store sum of weights after each measured step
    num_alive_walkers = []                                  # List to store the number of alive walkers at each step
    all_x_positions = []                                    # List to store positions for all measurement steps
    all_y_positions = []
    all_z_positions = []

    step_number = 0                                         # Step number counter

    # Run the simulation for the thermalising steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_thermalize, \
                          all_x_positions, all_y_positions, all_z_positions, num_alive_walkers, \
                          max_duplicates, 'thermalize', step_number)

    # Run the simulation for the measuring steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_measure, \
                          all_x_positions, all_y_positions, all_z_positions, num_alive_walkers, \
                          max_duplicates, 'measure', step_number)

    return all_x_positions, all_y_positions, all_z_positions, E_T_energies, w_E_L, sum_weights, num_alive_walkers


if __name__ == "__main__":
    N_0 = 10000                    # Initial number of alive walkers
    N_max = int(1.25 * N_0)        # Length of array
    num_steps_thermalize = 2500    # Number of steps to thermalise the system
    num_steps_measure = 10000      # Number of steps for which measurements are taken
    max_duplicates = 10            # Maximum number of duplicates produced during branching step
    delta_tau = 0.01               # Time step size
    alpha = 1/delta_tau            # Degree of feedback
    n_bins = 1000                  # Number of data bins for wavefunction histogram
    n_wait = int(0.1*num_steps_measure)   # Number of steps to wait before showing energy standard deviation on energy plot

    # Define the potential and trial function to be used
    potential_symbolic = -1/ (x**2 + y**2 + z**2)**0.5
    trial_psi_symbolic = (15-sp.sqrt(x**2 + y**2 + z**2))**12/(10**14)
    r = np.linspace(0, 8, n_bins)
    def trial_psi_r(r):
        return (15-r)**12/(10**14)

    # Corresponding velocity and local energy functions
    velocity_x_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, x) / trial_psi_symbolic )
    velocity_y_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, y) / trial_psi_symbolic )
    velocity_z_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, z) / trial_psi_symbolic )
    local_energy_symbolic = sp.simplify( -0.5 * (sp.diff(trial_psi_symbolic, x, 2) + sp.diff(trial_psi_symbolic, y, 2) + sp.diff(trial_psi_symbolic, z, 2)) / trial_psi_symbolic + potential_symbolic)
    # Convert symbolic expressions to functions
    potential = sp.lambdify([x,y,z], potential_symbolic)
    trial_psi = sp.lambdify([x,y,z], trial_psi_symbolic)
    velocity_x = sp.lambdify([x,y,z], velocity_x_symbolic)
    velocity_y = sp.lambdify([x,y,z], velocity_y_symbolic)
    velocity_z = sp.lambdify([x,y,z], velocity_z_symbolic)
    local_energy = sp.lambdify([x,y,z], local_energy_symbolic)

    measurements_x_positions, measurements_y_positions, measurements_z_positions, \
                              E_T_energies, w_E_L, sum_weights, num_alive_walkers \
                              = diffusion_monte_carlo(N_0, N_max, num_steps_thermalize, \
                                                        num_steps_measure, max_duplicates, delta_tau)

    # Generate the exact ground state wavefunction
    psi_R = np.exp(-r)
    Area_int_R = np.trapz((psi_R)**2 * r**2, r)
    Normalised_psi_R = psi_R / np.sqrt(Area_int_R)

    plt.figure()
    measurements_r_positions = np.sqrt(np.array(measurements_x_positions)**2+np.array(measurements_y_positions)**2+np.array(measurements_z_positions)**2)
    f_hist_values_r, f_bin_edges_r = np.histogram(measurements_r_positions, bins=n_bins, range=[0, 8], density=True)
    wavefunction_hist_values_r = f_hist_values_r / trial_psi_r(r)
    wavefunction_hist_values_r = (wavefunction_hist_values_r) / np.sqrt(np.sum(r[1:]**(-2)*(wavefunction_hist_values_r[1:])**2 * 8/n_bins))
    plt.plot(r[1:], r[1:]**(-2)*wavefunction_hist_values_r[1:], color='tab:pink', marker='x', ls='-', markersize = 4, label='DMC')
    plt.plot(r, Normalised_psi_R, color='tab:blue', label='Exact Wavefunction')
    plt.xlabel('r')
    plt.ylabel(r'Wavefunction, $\Psi$(r)')
    plt.legend()
    plt.title(r'H atom Ground State $\Psi$(r)')
    plt.savefig('Wavefunction.png')

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
    # Calculate the standard deviation of energy from measured steps
    running_std_energy = []
    for i in range(n_wait,len(E_L_energies)):
        running_std_energy.append(np.std(E_L_energies[:i]))
    std_energy = np.std(E_L_energies)
    mean_std_energy = std_energy
    print("Final Mean Energy from all Measurements:", final_mean_energy)
    print("Standard Deviation of Energy Mean from Measured Steps:", mean_std_energy)
    print("Exact Energy: -0.5")

     # Plot the energy variation with step number
    plt.figure()
    plt.plot(np.arange(len(E_L_energies)), E_L_energies, color='tab:green')
    upper_limit = np.array(E_L_energies[n_wait:]) + np.array(running_std_energy)
    lower_limit = np.array(E_L_energies[n_wait:]) - np.array(running_std_energy)
    plt.plot(np.arange(n_wait,len(E_L_energies)), upper_limit, color='tab:olive')
    plt.plot(np.arange(n_wait,len(E_L_energies)), lower_limit, color='tab:olive')
    plt.fill_between(np.arange(n_wait,len(E_L_energies)), lower_limit, upper_limit, color='tab:olive', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel(r'Energy /$E_H$')
    plt.title('Variation of Energy with Step Number')
    plt.axhline(y=-0.5, color='tab:red', linestyle='dashed', label='Exact Energy')
    plt.legend()
    plt.savefig('Energy.png')

    # Plot the number of alive walkers against step number
    plt.figure()
    plt.plot(np.arange(len(num_alive_walkers)), num_alive_walkers, color='tab:cyan')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Alive Walkers')
    plt.title('Variation of Number of Alive Walkers with Step Number')
    plt.savefig('Alive_walkers.png')
