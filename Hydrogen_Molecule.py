import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy
from scipy.stats import multivariate_normal


# Defining symbols to be used
x_n,y_n,z_n,x_n1,y_n1,z_n1,x_n2,y_n2,z_n2,x_e,y_e,z_e,x_e1,y_e1,z_e1,x_e2,y_e2,z_e2, x, y, z, r = \
    sp.symbols('x_n,y_n,z_n,x_n1,y_n1,z_n1,x_n2,y_n2,z_n2,x_e,y_e,z_e,x_e1,y_e1,z_e1,x_e2,y_e2,z_e2, x, y, z, r')


def G_d(z_old_n1, z_old_n2, x_old_e,y_old_e,z_old_e, x_new_e,y_new_e,z_new_e, delta_tau):
    """
    Importance sampling diffusive displacement probability density distribution for electrons. 
    The prefactor has been removed as we only require its relative value.
    This corresponds to Eqn 3.49 in Foulkes (2001).
    The value corresponds to the product of the probability densities for the independent a, y and z displacements.

        Parameters:
            z_old_n1[/2] (float): The original z position of nucleus 1[/2]
            x[/y/z]_old_e (float): The original x[/y/z] position of an electron
            x[/y/z]_new_e (float): The new x[/y/z] position of an electron
            delta_tau (float): The time incrament

        Returns:
            Probability density (float): The probability density of an electron moving from the specified old coordinates
                                         to the specified new coordinates
    """
    return np.exp(-((x_new_e - x_old_e - delta_tau * velocity_x(z_old_n1, z_old_n2, x_old_e,y_old_e,z_old_e))**2) / (2*delta_tau)) * \
           np.exp(-((y_new_e - y_old_e - delta_tau * velocity_y(z_old_n1, z_old_n2, x_old_e,y_old_e,z_old_e))**2) / (2*delta_tau)) * \
           np.exp(-((z_new_e - z_old_e - delta_tau * velocity_z(z_old_n1, z_old_n2, x_old_e,y_old_e,z_old_e))**2) / (2*delta_tau))


def G_d_nuclei(z_old_n, z_new_n, delta_tau):
    """
    Importance sampling diffusive displacement probability density distribution for nuclei. 
    The prefactor has been removed as we only require its relative value.
    This corresponds to Eqn 3.49 in Foulkes (2001).

        Parameters:
            z_old_n (float): The original z position of the nucleus
            z_new_n (float): The new z position of the nucleus
            delta_tau (float): The time incrament

        Returns:
            Probability density (float): The probability density of the nucleus moving from the specified old coordinates
                                         to the specified new coordinates
    """
    return np.exp(-(1836*(z_new_n - z_old_n - delta_tau * Nuclear_velocity_z(z_old_n)/1836)**2) / (2*delta_tau))


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
    return np.exp(-delta_tau * (new_local_energies + old_local_energies - 2*energy_offset) / 2)


def walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps, \
                          all_z_positions_n1, all_z_positions_n2, \
                          all_x_positions_e1, all_y_positions_e1, all_z_positions_e1, \
                          all_x_positions_e2, all_y_positions_e2, all_z_positions_e2, \
                          num_alive_walkers, max_duplicates, name, step_number, delta_tau, alpha):
    """
    Performs the walking and branching steps of the Diffusion Monte Carlo process.

        Parameters:
            walkers (float): A (4 x N_max x 4) array containing the walkers initial positions and alive/dead status 
            E_T_energies (float): An empty array
            w_E_L (float): An empty array
            sum_weights (float): An initially empty array that is appended with the value of the sum of the weights
            num_steps (int): Number of steps to run the calculation for
            all_x[/y/z]_positions_e1[/e2/n1/n2] (float): An empty array
            num_alive_walkers (float): An empty array
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step
            name (str): 'thermalize' or 'measure' indicates if the current steps are thermalising the system or being measured
            step_number (int): A counter to keep track of the current step number
            delta_tau (float): The time incrament
            alpha (float): The degree of feedback

        Returns:
            walkers (float): A (4 x N_max x 4) array containing the walkers final positions and alive/dead status 
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            all_x[/y/z]_positions_e1[/e2/n1/n2] (float): An array containing the x[/y/z] position of e1[/e2/n1/n2] after each
                                                         measured step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
            step_number (int): Final step counter
    """

    # For each step, perform walking and branching of the walkers
    for step in range(num_steps):
        # Calculate the local energy for the given walker positions
        old_local_energies = local_energy(walkers[3, (walkers[0].T)[0] == 1, 0], walkers[3, (walkers[0].T)[0] == 1, 1], \
                                          walkers[1, (walkers[0].T)[0] == 1, 2], walkers[2, (walkers[0].T)[0] == 1, 2], \
                                          walkers[3, (walkers[0].T)[0] == 1, 2]) + \
                              local_energy(walkers[3, (walkers[0].T)[0] == 1, 0], walkers[3, (walkers[0].T)[0] == 1, 1], \
                                           walkers[1, (walkers[0].T)[0] == 1, 3], walkers[2, (walkers[0].T)[0] == 1, 3], \
                                           walkers[3, (walkers[0].T)[0] == 1, 3]) + \
                              1/np.sqrt((walkers[3, (walkers[0].T)[0] == 1, 0]-walkers[3, (walkers[0].T)[0] == 1, 1])**2) + \
                              1/np.sqrt((walkers[1, (walkers[0].T)[0] == 1, 2]-walkers[1, (walkers[0].T)[0] == 1, 3])**2 + \
                                        (walkers[2, (walkers[0].T)[0] == 1, 2]-walkers[2, (walkers[0].T)[0] == 1, 3])**2 + \
                                        (walkers[3, (walkers[0].T)[0] == 1, 2]-walkers[3, (walkers[0].T)[0] == 1, 3])**2)

        # Move the walkers by a random amount, here all walkers are moved at once in a given step
        z_old_n1 = walkers[3, :, 0]
        z_proposed_n1 = z_old_n1 + np.random.normal(scale=np.sqrt(delta_tau/1836), size=N_max) + delta_tau * Nuclear_velocity_z(z_old_n1)/1836
        z_old_n2 = walkers[3, :, 1]
        z_proposed_n2 = z_old_n2 + np.random.normal(scale=np.sqrt(delta_tau/1836), size=N_max) + delta_tau * Nuclear_velocity_z(z_old_n2)/1836
        x_old_e1 = walkers[1, :, 2]
        y_old_e1 = walkers[2, :, 2]
        z_old_e1 = walkers[3, :, 2]
        x_proposed_e1 = x_old_e1 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_x(z_old_n1, z_old_n2, x_old_e1,y_old_e1,z_old_e1)
        y_proposed_e1 = y_old_e1 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_y(z_old_n1, z_old_n2, x_old_e1,y_old_e1,z_old_e1)
        z_proposed_e1 = z_old_e1 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_z(z_old_n1, z_old_n2, x_old_e1,y_old_e1,z_old_e1)
        x_old_e2 = walkers[1, :, 3]
        y_old_e2 = walkers[2, :, 3]
        z_old_e2 = walkers[3, :, 3]
        x_proposed_e2 = x_old_e2 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_x(z_old_n1, z_old_n2, x_old_e2,y_old_e2,z_old_e2)
        y_proposed_e2 = y_old_e2 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_y(z_old_n1, z_old_n2, x_old_e2,y_old_e2,z_old_e2)
        z_proposed_e2 = z_old_e2 + np.random.normal(scale=np.sqrt(delta_tau), size=N_max) + \
                        delta_tau * velocity_z(z_old_n1, z_old_n2, x_old_e2,y_old_e2,z_old_e2)

        # Iterating over each of the walkers
        for i in range(len(walkers[1])):
            # Only if the walkers are alive
            if walkers[0, i, 0] == 1:
                # Calculate p, the probability of accepting the steps taken by the walker. Formula given by Foulkes (2001) Eqn 3.52
                p = np.minimum(1, (G_d(z_proposed_n1[i], z_proposed_n2[i], x_proposed_e1[i],y_proposed_e1[i],z_proposed_e1[i], \
                                       x_old_e1[i],y_old_e1[i],z_old_e1[i], delta_tau) * \
                                       (trial_psi(z_proposed_n1[i],z_proposed_n2[i],x_proposed_e1[i],y_proposed_e1[i],z_proposed_e1[i]))**2) \
                                / (G_d(z_old_n1[i], z_old_n2[i], x_old_e1[i],y_old_e1[i],z_old_e1[i], \
                                       x_proposed_e1[i],y_proposed_e1[i],z_proposed_e1[i], delta_tau) * \
                                       (trial_psi(z_old_n1[i],z_old_n2[i],x_old_e1[i],y_old_e1[i],z_old_e1[i]))**2) \
                                * (G_d(z_proposed_n1[i], z_proposed_n2[i], x_proposed_e2[i],y_proposed_e2[i],z_proposed_e2[i], \
                                       x_old_e2[i],y_old_e2[i],z_old_e2[i], delta_tau) * \
                                       (trial_psi(z_proposed_n1[i],z_proposed_n2[i],x_proposed_e2[i],y_proposed_e2[i],z_proposed_e2[i]))**2) \
                                / (G_d(z_old_n1[i], z_old_n2[i], x_old_e2[i],y_old_e2[i],z_old_e2[i], \
                                       x_proposed_e2[i],y_proposed_e2[i],z_proposed_e2[i], delta_tau) * \
                                       (trial_psi(z_old_n1[i],z_old_n2[i],x_old_e2[i],y_old_e2[i],z_old_e2[i]))**2) \
                                * (G_d_nuclei(z_proposed_n1[i], z_old_n1[i], delta_tau) * (Nuclear_trial_psi(z_proposed_n1[i]))**2) \
                                / (G_d_nuclei(z_old_n1[i], z_proposed_n1[i], delta_tau) * (Nuclear_trial_psi(z_old_n1[i]))**2) \
                                * (G_d_nuclei(z_proposed_n2[i], z_old_n2[i], delta_tau) * (Nuclear_trial_psi(z_proposed_n2[i]))**2) \
                                / (G_d_nuclei(z_old_n2[i], z_proposed_n2[i], delta_tau) * (Nuclear_trial_psi(z_old_n2[i]))**2) )
                # Metropolis acceptance criterion
                if p == 1:
                    walkers[3, i, 0] = z_proposed_n1[i]
                    walkers[3, i, 1] = z_proposed_n2[i]
                    walkers[1, i, 2] = x_proposed_e1[i]
                    walkers[2, i, 2] = y_proposed_e1[i]
                    walkers[3, i, 2] = z_proposed_e1[i]
                    walkers[1, i, 3] = x_proposed_e2[i]
                    walkers[2, i, 3] = y_proposed_e2[i]
                    walkers[3, i, 3] = z_proposed_e2[i]
                else:
                    if np.random.uniform(0, 1) <= p:
                        walkers[3, i, 0] = z_proposed_n1[i]
                        walkers[3, i, 1] = z_proposed_n2[i]
                        walkers[1, i, 2] = x_proposed_e1[i]
                        walkers[2, i, 2] = y_proposed_e1[i]
                        walkers[3, i, 2] = z_proposed_e1[i]
                        walkers[1, i, 3] = x_proposed_e2[i]
                        walkers[2, i, 3] = y_proposed_e2[i]
                        walkers[3, i, 3] = z_proposed_e2[i]
                    else:
                        walkers[3, i, 0] = z_old_n1[i]
                        walkers[3, i, 1] = z_old_n2[i]
                        walkers[1, i, 2] = x_old_e1[i]
                        walkers[2, i, 2] = y_old_e1[i]
                        walkers[3, i, 2] = z_old_e1[i]
                        walkers[1, i, 3] = x_old_e2[i]
                        walkers[2, i, 3] = y_old_e2[i]
                        walkers[3, i, 3] = z_old_e2[i]

        # Calculate the local energy for the new walker positions
        new_local_energies = local_energy(walkers[3, (walkers[0].T)[0] == 1, 0], walkers[3, (walkers[0].T)[0] == 1, 1], \
                                          walkers[1, (walkers[0].T)[0] == 1, 2], walkers[2, (walkers[0].T)[0] == 1, 2], \
                                          walkers[3, (walkers[0].T)[0] == 1, 2]) + \
                              local_energy(walkers[3, (walkers[0].T)[0] == 1, 0], walkers[3, (walkers[0].T)[0] == 1, 1], \
                                           walkers[1, (walkers[0].T)[0] == 1, 3], walkers[2, (walkers[0].T)[0] == 1, 3], \
                                           walkers[3, (walkers[0].T)[0] == 1, 3]) + \
                              1/np.sqrt((walkers[3, (walkers[0].T)[0] == 1, 0]-walkers[3, (walkers[0].T)[0] == 1, 1])**2) + \
                              1/np.sqrt((walkers[1, (walkers[0].T)[0] == 1, 2]-walkers[1, (walkers[0].T)[0] == 1, 3])**2 + \
                                        (walkers[2, (walkers[0].T)[0] == 1, 2]-walkers[2, (walkers[0].T)[0] == 1, 3])**2 + \
                                        (walkers[3, (walkers[0].T)[0] == 1, 2]-walkers[3, (walkers[0].T)[0] == 1, 3])**2)

        # Count the number of walkers that are alive at this step and store the number
        N_alive = sum(walkers[0, :, 0])
        num_alive_walkers.append(N_alive)

        # Calculate the reference/ offset energy at this step and store the number
        E_T = np.mean(potential(walkers[3, (walkers[0].T)[0] == 1, 0], \
                                walkers[3, (walkers[0].T)[0] == 1, 1], \
                                walkers[1, (walkers[0].T)[0] == 1, 2], walkers[2, (walkers[0].T)[0] == 1, 2], walkers[3, (walkers[0].T)[0] == 1, 2]) + \
                      potential(walkers[3, (walkers[0].T)[0] == 1, 0], \
                                walkers[3, (walkers[0].T)[0] == 1, 1], \
                                walkers[1, (walkers[0].T)[0] == 1, 3], walkers[2, (walkers[0].T)[0] == 1, 3], walkers[3, (walkers[0].T)[0] == 1, 3]) + \
                      1/np.sqrt((walkers[3, (walkers[0].T)[0] == 1, 0]-walkers[3, (walkers[0].T)[0] == 1, 1])**2) + \
                      1/np.sqrt((walkers[1, (walkers[0].T)[0] == 1, 2]-walkers[1, (walkers[0].T)[0] == 1, 3])**2 + \
                                (walkers[2, (walkers[0].T)[0] == 1, 2]-walkers[2, (walkers[0].T)[0] == 1, 3])**2 + \
                                (walkers[3, (walkers[0].T)[0] == 1, 2]-walkers[3, (walkers[0].T)[0] == 1, 3])**2)) + \
                      alpha * (1 - N_alive / N_0)
        E_T_energies.append(E_T)

        # Calculate the weights of each walker
        weights = importance_sampling_weight(old_local_energies, new_local_energies, E_T, delta_tau)

        # Calculate the duplicity of each walker
        m_ns = np.minimum(np.floor(weights + np.random.rand(len(walkers[0, (walkers[0].T)[0] == 1, 0]))), max_duplicates + 1)  # Eqn 2.28

        # Find the last alive walker to add on replicated walkers after this
        N_last_alive = N_max - np.argmax((np.fliplr(walkers[:,:,0])[0, :]) == 1) - 1

        # If the array is nearly filled, make a new walkers array only containing current alive walkers and extra space to fill
        if N_last_alive >= 0.95 * N_max:
            number_of_alive = len(walkers[0, (walkers[0].T)[0] == 1, 0])
            walkers[:, :number_of_alive] = walkers[:, (walkers[0].T)[0] == 1, :]
            walkers[0, :number_of_alive] = 1
            walkers[0, number_of_alive:] = 0
            #print('RESET')

        # Find the last alive walker of the new array to add on replicated walkers after this
        N_last_alive = N_max - np.argmax((np.fliplr(walkers[:,:,0])[0, :]) == 1) - 1

        # Duplicate/ kill walkers
        # Define a counter to keep track of the alive walkers
        counter = -1
        # Iterating over all the alive walkers, including the dead ones in between
        for i in range(N_last_alive):
            # If walker is alive
            if walkers[0, i, 0] == 1:
                # Increase counter by 1 to correctly iterate through the weights array
                counter += 1
                if m_ns[counter] == 0:
                    # If duplicity is 0, kill walker
                    walkers[0, i, 0] = 0
                elif 2 <= m_ns[counter] <= int(max_duplicates + 1):
                    # Else, duplicate the walker
                    for j in range(int(m_ns[counter]) - 1):
                        N_last_alive += 1
                        walkers[:, N_last_alive, :] = walkers[:, i, :]

        # For measured steps
        if str(name) == 'measure':
            # Store positions of all alive walkers
            all_z_positions_n1.extend(walkers[3, (walkers[0].T)[0] == 1, 0])
            all_z_positions_n2.extend(walkers[3, (walkers[0].T)[0] == 1, 1])
            all_x_positions_e1.extend(walkers[1, (walkers[0].T)[0] == 1, 2])
            all_y_positions_e1.extend(walkers[2, (walkers[0].T)[0] == 1, 2])
            all_z_positions_e1.extend(walkers[3, (walkers[0].T)[0] == 1, 2])
            all_x_positions_e2.extend(walkers[1, (walkers[0].T)[0] == 1, 3])
            all_y_positions_e2.extend(walkers[2, (walkers[0].T)[0] == 1, 3])
            all_z_positions_e2.extend(walkers[3, (walkers[0].T)[0] == 1, 3])
            # Calculate and store the value of the sum of the product of the weights and local energies, as well as the sum
            # of weights
            alive_w_E_L = []
            alive_sum_weights = []
            for i in range(len(new_local_energies)):
                if walkers[0, i, 0] == 1:
                    alive_w_E_L.append(weights[i] * new_local_energies[i])
                    alive_sum_weights.append(weights[i])
            w_E_L.append(np.sum(alive_w_E_L))
            sum_weights.append(np.sum(alive_sum_weights))

        # Increase step counter by 1
        step_number += 1
        #print(step_number)

    return walkers, E_T_energies, w_E_L, sum_weights, all_z_positions_n1, all_z_positions_n2, all_x_positions_e1, all_y_positions_e1, all_z_positions_e1, all_x_positions_e2, all_y_positions_e2, all_z_positions_e2, num_alive_walkers, step_number


def diffusion_monte_carlo(N_0, N_max, num_steps_thermalize_n, num_steps_measure_n, max_duplicates):
    """
    Perform Diffusion Monte Carlo for the H2 system. Sets up the initial system with guessed starting positions for the walkers,
    runs the thermalising then measuring, walking and branching steps.

        Parameters:
            N_0 (int): Initial number of alive walkers
            N_max (int): Length of array
            num_steps_thermalize_n (int): Number of steps to thermalise the system
            num_steps_measure_n (int): Number of measured steps
            max_duplicates (int): The maximum number of duplicate walkers that can arise from a given walker in a given step

        Returns:
            all_x[/y/z]_positions_e1[/e2/n1/n2] (float): An array containing the x[/y/z] position of e1[/e2/n1/n2] after each
                                                         measured step
            E_T_energies (float): An array containing the value of the energy offset for each step
            w_E_L (float): An array containing the value of the sum of the product of the weight and local energy
                           after each step
            sum_weights (float): An array of the sum of the weights for each step
            num_alive_walkers (float): An array containing the number of alive walkers after each step
    """

    # Set up 3D array of walkers
    # The first dimension is indexed by 'alive/ dead status', 'x-position', 'y-position', 'z-position'
    # The second dimension is indexed by the walker number
    # The third dimension is indexed by the particle 'nucleus 1', 'nucleus 2', 'electron 1', 'electron 2'
    walkers = np.zeros((4, N_max, 4))
    # Set the first N_0 walkers to be alive
    walkers[0, :N_0, 0] = 1
    # Generate positions for the nuclei, using the nuclear trial wavefunction as the probability distribution
    n_probs=100*N_max
    x_n1_samples = np.random.uniform(0, 0, n_probs)
    y_n1_samples = np.random.uniform(0, 0, n_probs)
    z_n1_samples = np.random.uniform(-1.4, -0.2, n_probs)
    probability_values_n1 = (Nuclear_trial_psi(z_n1_samples))**2
    sum_n1 = np.sum(probability_values_n1)
    probability_values_n1 = probability_values_n1/sum_n1
    walkers[1, :, 0] = np.zeros(N_max)
    walkers[2, :, 0] = np.zeros(N_max)
    walkers[3, :, 0] = np.random.choice(z_n1_samples, N_max, p=probability_values_n1)
    # Set the second nuclei positions to be an inversion of the first nucleus
    walkers[1, :, 1] = np.zeros(N_max)
    walkers[2, :, 1] = np.zeros(N_max)
    walkers[3, :, 1] = - walkers[3, :, 0]

    # Choosing reasonable average nuclear positions, generate electron positions using the electron trial
    # wavefunction as the probability distribution
    z_n1 = -0.7
    z_n2 = 0.7
    x_e1_samples = np.random.uniform(-5, 5, n_probs)
    y_e1_samples = np.random.uniform(-5, 5, n_probs)
    z_e1_samples = np.random.uniform(-5, 5, n_probs)
    probability_values_e1 = (trial_psi(z_n1,z_n2, x_e1_samples, y_e1_samples, z_e1_samples))**2
    sum_p1 = np.sum(probability_values_e1)
    probability_values_e1 = probability_values_e1/sum_p1
    walkers[1, :, 2] = np.random.choice(x_e1_samples, N_max, p=probability_values_e1)
    walkers[2, :, 2] = np.random.choice(y_e1_samples, N_max, p=probability_values_e1)
    walkers[3, :, 2] = np.random.choice(z_e1_samples, N_max, p=probability_values_e1)
    # Let the second electron position be independent of the first, so select from the same distribution as before
    x_e2_samples = np.random.uniform(-5, 5, n_probs)
    y_e2_samples = np.random.uniform(-5, 5, n_probs)
    z_e2_samples = np.random.uniform(-5, 5, n_probs)
    probability_values_e2 = (trial_psi(z_n1,z_n2, x_e2_samples, y_e2_samples, z_e2_samples))**2
    sum_p2 = np.sum(probability_values_e2)
    probability_values_e2 = probability_values_e2/sum_p2
    walkers[1, :, 3] = np.random.choice(x_e2_samples, N_max, p=probability_values_e2)
    walkers[2, :, 3] = np.random.choice(y_e2_samples, N_max, p=probability_values_e2)
    walkers[3, :, 3] = np.random.choice(z_e2_samples, N_max, p=probability_values_e2)

    E_T_energies = []           # Array to store energy after each step
    w_E_L = []                  # Array to store sum of product of weight and local energy after each measured step
    sum_weights = []            # Array to store sum of weights after each measured step
    num_alive_walkers = []      # List to store the number of alive walkers at each step
    all_z_positions_n1 = []     # Lists to store positions of all particles for all measurement steps
    all_z_positions_n2 = []
    all_x_positions_e1 = []
    all_y_positions_e1 = []
    all_z_positions_e1 = []
    all_x_positions_e2 = [] 
    all_y_positions_e2 = []
    all_z_positions_e2 = []
    step_number = 0             # Step number counter
    delta_tau = dt              # In general the time incrament can be changed, here set to single specified value
    alpha = alpha_0             # Corresponding degree of feedback

    # Run the simulation for the thermalising steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_thermalize_n, \
                          all_z_positions_n1, all_z_positions_n2, all_x_positions_e1, all_y_positions_e1, all_z_positions_e1, \
                          all_x_positions_e2, all_y_positions_e2, all_z_positions_e2, num_alive_walkers, max_duplicates, \
                          'thermalize', step_number, delta_tau, alpha)

    # Run the simulation for the measuring steps
    walking_and_branching(walkers, E_T_energies, w_E_L, sum_weights, num_steps_measure_n, \
                          all_z_positions_n1, all_z_positions_n2, all_x_positions_e1, all_y_positions_e1, all_z_positions_e1, \
                          all_x_positions_e2, all_y_positions_e2, all_z_positions_e2, num_alive_walkers, max_duplicates, \
                          'measure', step_number, delta_tau, alpha)

    return all_z_positions_n1, all_z_positions_n2, all_x_positions_e1, all_y_positions_e1, all_z_positions_e1, all_x_positions_e2, all_y_positions_e2, all_z_positions_e2, E_T_energies, w_E_L, sum_weights, num_alive_walkers


if __name__ == "__main__":
    N_0 = 1000                              # Initial number of alive walkers
    N_max = int(1.1 * N_0)                  # Length of array
    num_steps_thermalize_n = 150000         # Number of steps to thermalise the system
    num_steps_measure_n = 100000            # Number of steps for which measurements are taken
    max_duplicates = 3                      # Maximum number of duplicates produced during branching step
    dt = 0.001                              # Time incrament
    alpha_0 = 1/dt                          # Degree of feedback
    n_wait = int(0.1*num_steps_measure_n)   # Number of steps to wait before showing energy standard deviation on energy plot

    # Define the potential and trial function to be used
    potential_symbolic = -1/sp.sqrt((x_e)**2+(y_e)**2+(z_n1-z_e)**2) + \
                          -1/sp.sqrt((x_e)**2+(y_e)**2+(z_n2-z_e)**2)
    trial_psi_symbolic = sp.exp(-0.9*sp.sqrt((x_e)**2+(y_e)**2+(z_n1-z_e)**2)) + \
                          sp.exp(-0.9*sp.sqrt((x_e)**2+(y_e)**2+(z_n2-z_e)**2))

    # Corresponding velocity and local energy functions
    velocity_x_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, x_e) / trial_psi_symbolic )
    velocity_y_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, y_e) / trial_psi_symbolic )
    velocity_z_symbolic = sp.simplify( sp.diff(trial_psi_symbolic, z_e) / trial_psi_symbolic )
    local_energy_symbolic = -0.5 * (sp.diff(trial_psi_symbolic, x_e, 2) + sp.diff(trial_psi_symbolic, y_e, 2) + sp.diff(trial_psi_symbolic, z_e, 2)) / trial_psi_symbolic + potential_symbolic
    # Convert symbolic expressions to functions
    potential = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], potential_symbolic)
    trial_psi = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], trial_psi_symbolic)
    velocity_x = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], velocity_x_symbolic)
    velocity_y = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], velocity_y_symbolic)
    velocity_z = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], velocity_z_symbolic)
    local_energy = sp.lambdify([z_n1,z_n2,x_e,y_e,z_e], local_energy_symbolic)

    # Define nuclear trial function
    Nuclear_trial_psi_symbolic = z_n**2 * sp.exp(-2*(z_n**2))
    # Corresponding velocity and local energy functions must be calculated
    Nuclear_velocity_z_symbolic = sp.diff(Nuclear_trial_psi_symbolic, z_n) / Nuclear_trial_psi_symbolic
    Nuclear_local_energy_symbolic = -0.5 * (sp.diff(Nuclear_trial_psi_symbolic, z_n, 2)) / Nuclear_trial_psi_symbolic
    Nuclear_trial_psi = sp.lambdify([z_n], Nuclear_trial_psi_symbolic)
    Nuclear_velocity_z = sp.lambdify([z_n], Nuclear_velocity_z_symbolic)
    Nuclear_local_energy = sp.lambdify([z_n], Nuclear_local_energy_symbolic)

    # Run Diffusion Monte Carlo 
    measurements_z_positions_n1, measurements_z_positions_n2, measurements_x_positions_e1, \
                                 measurements_y_positions_e1, measurements_z_positions_e1, \
                                 measurements_x_positions_e2, measurements_y_positions_e2, \
                                 measurements_z_positions_e2, E_T_energies, w_E_L, sum_weights, \
                                 num_alive_walkers = diffusion_monte_carlo(N_0, N_max, num_steps_thermalize_n, \
                                                                           num_steps_measure_n, max_duplicates)

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
    print("Final Mean Energy from all Measurements:", final_mean_energy)
    print("Standard Deviation of Energy Mean from Measured Steps:", mean_std_energy)
    print("Eqm energy with ZPE!!: -1.1645")

    # Plot the energy variation with step number
    plt.figure()
    plt.plot(np.arange(len(E_L_energies)), E_L_energies, color='tab:green')
    upper_limit = np.array(E_L_energies[n_wait:]) + np.array(running_std_energy)
    lower_limit = np.array(E_L_energies[n_wait:]) - np.array(running_std_energy)
    plt.plot(np.arange(n_wait,len(E_L_energies)), upper_limit, color='tab:olive')
    plt.plot(np.arange(n_wait,len(E_L_energies)), lower_limit, color='tab:olive')
    plt.fill_between(np.arange(n_wait,len(E_L_energies)), lower_limit, upper_limit, color='tab:olive', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel(r'Energy / Hartree')
    plt.title('Variation of Energy with Step Number')
    plt.axhline(y=-1.1645, color='tab:red', linestyle='dashed', label='Exact Energy')
    plt.legend()
    plt.savefig('Energy.png', format='png')

    # Plot the number of alive walkers against step number
    plt.figure()
    plt.plot(np.arange(len(num_alive_walkers)), num_alive_walkers, color='tab:cyan')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Alive Walkers')
    plt.title('Variation of Number of Alive Walkers with Step Number')
    plt.savefig('Alive_Walkers.png', format='png')

    # Calculate the bond legnths 
    bond_lengths = np.array(measurements_z_positions_n2)-np.array(measurements_z_positions_n1)
    plt.figure()
    n_bins = 100
    f_hist_values, f_bin_edges = np.histogram(bond_lengths, bins=n_bins, range=[0.5, 2.5], density=True)
    z_n_list = np.zeros(shape=n_bins)
    for i in range(n_bins):
        z_n_list[i] = (f_bin_edges[i]+f_bin_edges[i+1])/2
    wavefunction_hist_values = f_hist_values / ( (z_n_list/2)**2 * np.exp(-2*(z_n_list/2)**2) )
    wavefunction_hist_values = (wavefunction_hist_values) / np.sqrt(np.sum(wavefunction_hist_values**2 * 2/n_bins))
    plt.plot(z_n_list, f_hist_values, color='tab:blue', marker='x', ls='-', markersize = 4, label='DMC')
    plt.xlabel('Bond Length')
    plt.ylabel(r'N of walkers')
    plt.legend()
    plt.title('Bond Length')
    plt.savefig('Bond_Histogram.png', format='png')

    # Calculate the distance between the electrons
    r12 = np.linspace(-0.1, 6, n_bins)
    plt.figure()
    measurements_r12_positions = np.sqrt((np.array(measurements_x_positions_e1)-np.array(measurements_x_positions_e2))**2 + \
                                         (np.array(measurements_y_positions_e1)-np.array(measurements_y_positions_e2))**2 + \
                                         (np.array(measurements_z_positions_e1)-np.array(measurements_z_positions_e2))**2 )
    f_hist_values_r12, f_bin_edges_r12 = np.histogram(measurements_r12_positions, bins=n_bins, range=[-0.1, 6], density=True)
    wavefunction_hist_values_r12 = f_hist_values_r12
    wavefunction_hist_values_r12 = (wavefunction_hist_values_r12) / np.sqrt(np.sum((wavefunction_hist_values_r12) * 6.1/n_bins))
    plt.plot(r12, wavefunction_hist_values_r12, color='tab:gray', marker='x', ls='-', markersize = 4)
    plt.xlabel(r'Inter-electron distance, $r_{12}$')
    plt.ylabel(r'Probability')
    plt.title(r'Inter-electron distance')
    plt.savefig('Interelectron_Distance.png', format='png')

    # Plot the reference energy with step number
    plt.figure()
    plt.plot(np.arange(len(E_T_energies)),E_T_energies)
    plt.xlabel('Time Step$')
    plt.ylabel(r'Energy / Hartree')
    plt.title('Offset Energy')
    plt.savefig('E_T_Energies.png', format='png')
    
