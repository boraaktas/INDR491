import math
import random
import time
from datetime import datetime
from typing import Callable, Union, Optional

from matplotlib import pyplot as plt


def TS(init_method: Callable,
       N_List: list[Callable],
       objective_method: Callable,
       tabu_size: int,
       num_neighbors: int,
       N_func_weights: Union[str, list[Union[float, int]]] = 'uniform',
       reaction_factor: float = 1,
       time_limit: float = 1000,
       ITER: int = 1000000,
       benchmark_cost: float = 0,
       benchmark_opt_bool: bool = False,
       threshold_gap: float = 0.01,
       print_iteration: bool = False,
       print_results: bool = False,
       plot_results: bool = False,
       save_results: bool = False,
       save_results_path: str = '',
       save_results_file_name: str = '') -> list:
    """
    Tabu Search Algorithm
    """

    def check_termination_conditions(iteration: int, ITER: int, start_time: float, time_limit: float) -> bool:
        """
        Check if the termination conditions are met.
        The termination conditions are:
            - The number of iterations exceeds the maximum number of iterations
            - The elapsed time exceeds the maximum time
            ...

        Args:
            iteration (int): number of iterations
            ITER (int): maximum number of iterations
            start_time (float): time at which the algorithm started
            time_limit (float): maximum time allowed for the algorithm to run

        Returns:
            bool: True if the termination conditions are met, False otherwise
        """

        if iteration >= ITER:
            return True

        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            return True

        return False

    def generate_neighbors(N_List, N_func_probs, S_current, num_neighbors):
        """
        Generate neighbors of the current solution and their moves (the moves are stored in a dictionary)
        by applying the neighbor function from the list of neighbor functions to the current solution num_neighbors times

        Args:
            N_List (list): list of neighbor functions
            N_func_probs (list): list of probabilities of choosing each neighbor function
            S_current (list): current solution
            num_neighbors (int): number of neighbors to generate

        Returns:
            neighbors (list): list of neighbors of the current solution
            moves (list): list of moves that were made to get to the neighbors in the neighbors list with the same order
        """

        # neighbors is a dictionary of the form {neighbor: move}
        neighbors = []
        moves = []

        for i in range(num_neighbors):
            # Choose a random neighbor function from the list
            rand = random.choices(range(len(N_List)), weights=N_func_probs)[0]
            neighbor_func = N_List[rand]

            neighbor, move = neighbor_func(S_current)

            # add the neighbor and its move to the lists
            # neighbors is a list of list
            neighbors.append(neighbor)

            # moves is a list of tuples of the form ('function_name', move_tuple)
            moves.append(tuple((neighbor_func.__name__, move)))

        return neighbors, moves

    def check_tabu_list(move_tuple, tabu_list):
        """
        Check if the move is in the tabu list

        Args:
            move_tuple (tuple): move that was made to get to the solution
            tabu_list (list): list of solutions that are tabu

        Returns:
            bool: True if the move is in the tabu list, False otherwise
        """

        move_func = move_tuple[0]
        move = move_tuple[1]

        for tabu in tabu_list:
            # Check for first move function name
            if tabu[0] == move_func:
                # If the move function name is the same, check for the move
                if tabu[1] == move:
                    return True

        return False

    def change_function_probabilities(N_List, N_func_probs, winner_index, prize):
        # increase the probability of the winner function
        N_func_probs[winner_index] += prize

        # rearrange the probabilities to make sure that the sum of the probabilities is 1
        N_func_probs = [p / sum(N_func_probs) for p in N_func_probs]

        #
        if N_func_probs[winner_index] > 0.5:
            N_func_probs = [1 / len(N_List) for _ in range(len(N_List))]

        return N_func_probs

    def change_function_probabilities_2(func_probs: list[float],
                                        best_move_func_index: int,
                                        selected_solution_obj: float,
                                        best_solution_obj: float,
                                        current_solution_obj: float,
                                        react_factor: float) -> list[float]:
        """
        Change the probabilities of choosing each neighbor function based on the reaction factor and the prize
        from: https://doi.org/10.1016/j.trc.2019.02.018

        Args:
            func_probs (list): list of probabilities of choosing each neighbor function
            best_move_func_index (int): index of the best neighbor function
            selected_solution_obj (float): objective value of the selected solution
            best_solution_obj (float): objective value of the best solution found so far
            current_solution_obj (float): objective value of the current solution
            react_factor (float): reaction factor that is the weight of the old probabilities of each function
        Returns:
            N_func_probs (list): list of probabilities of choosing each neighbor function
        """

        # Calculate the prize
        if selected_solution_obj > best_solution_obj:
            prize = 0.001
        elif selected_solution_obj > current_solution_obj:
            prize = 0.0001
        else:
            prize = 0.0

        # update the weights of the neighbor functions
        for j in range(len(func_probs)):
            if j == best_move_func_index:
                func_probs[j] = react_factor * func_probs[j] + prize * (1 - reaction_factor)
            else:
                func_probs[j] = react_factor * func_probs[j]

        # rearrange the probabilities to make sure that the sum of the probabilities is 1
        func_probs = [prob / sum(func_probs) for prob in func_probs]

        return func_probs

    def check_aspiration_criteria(move, neighbor_cost, best_cost, tabu_list) -> tuple[bool, list]:
        """
        Check if the solution is in the tabu list and if it is,
        check if it is better than the best solution in the tabu list

        Args:
            move (tuple): move that was made to get to the neighbor solution
            neighbor_cost (float): cost of the neighbor solution
            best_cost (float): cost of the best solution found so far
            tabu_list (list): list of solutions that are tabu

        Returns: bool: True if the solution is in the tabu list, and it is better than the best solution in the tabu list,
                        False otherwise
        """

        # Check if the neighbor solution that is in the tabu list is better than the best solution found so far
        if neighbor_cost < best_cost:
            """
            print('Aspiration criteria met')
            print('Move selected from the tabu list: ', move)
            """
            # update the tabu list
            tabu_list.remove(move)
            return True, tabu_list

        return False, tabu_list

    if len(N_List) == 0:
        # If the list of neighbor functions is empty, raise an error
        msg = "The list of neighbor functions is empty."
        raise ValueError(msg)
    elif len(N_List) != len(N_func_weights) and isinstance(N_func_weights, list):
        # If the number of neighbor functions and the number of probabilities of choosing each neighbor function
        # are not the same, raise an error
        msg = "The number of neighbor functions and the number of probabilities of choosing each neighbor function " \
              "are not the same."
        raise ValueError(msg)

    if isinstance(N_func_weights, str):
        # If the probability distribution of choosing each neighbor function is not a list or 'uniform', raise an error
        if N_func_weights.lower() != 'uniform':
            msg = "The probability distribution of choosing each neighbor function is not valid." \
                  "It should be either 'uniform' or a list of weights."
            raise ValueError(msg)
    elif isinstance(N_func_weights, list):
        # check if all the elements in the list of weights are positive and integers or floats
        for p in N_func_weights:
            if not isinstance(p, (int, float)) or p < 0:
                msg = "The probability distribution of choosing each neighbor function is not valid." \
                      "It should be either 'uniform' or a list of positive weights."
                raise ValueError(msg)

    if reaction_factor < 0 or reaction_factor > 1:
        # If the reaction factor is not between 0 and 1, raise an error
        msg = "The reaction factor should be between 0 and 1."
        raise ValueError(msg)

    if save_results:
        if save_results_path == '':
            msg = "The path to save the results is empty."
            raise ValueError(msg)
        elif save_results_file_name == '':
            msg = "The file name to save the results is empty."
            raise ValueError(msg)

    # Number of neighbor functions
    no_functions: int = len(N_List)

    start_time: float = time.time()  # start time of the algorithm

    # Initialize the solution and its cost
    S_initial = init_method()
    cost_initial = objective_method(S_initial)

    # Initialize the current solution and its cost
    S_current = S_initial
    cost_current = cost_initial

    # Initialize the best solution and its cost as the current solution and its cost
    S_best = S_current
    cost_best = cost_current
    best_time = start_time
    best_iter = 0

    # Keep track of all the solutions found so far
    solutions = [S_current]
    solution_costs = [cost_current]

    # Initialize the tabu list
    tabu_list: list = []

    # Keep track of the number of times a tabu solution was chased and selected
    number_of_chased_tabu = 0

    # Keep track of the number of times a tabu solution met the aspiration criteria and was selected
    number_of_selected_tabu = 0

    # Initialize the probabilities of choosing each neighbor function
    N_func_probs: list[float] = [0 for _ in range(no_functions)]

    if isinstance(N_func_weights, str) and N_func_weights.lower() == 'uniform':
        N_func_probs = [1 / no_functions for _ in range(no_functions)]
    elif isinstance(N_func_weights, list):
        N_func_probs = [p / sum(N_func_weights) for p in N_func_weights]

    # Keep track of the number of times each neighbor function was the best neighbor selected in each iteration
    N_func_selected = [0 for _ in range(no_functions)]
    # Keep track of the number of times each neighbor function improved the best solution
    N_func_improves = [0 for _ in range(no_functions)]

    # Keep track of the gap between the current solution found so far and the benchmark cost
    gap: float = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
    threshold_gap_time: Optional[float] = None
    threshold_gap_iter: Optional[int] = None
    if gap <= threshold_gap:
        threshold_gap_time = time.time() - start_time
        threshold_gap_iter = 0

    iteration = 0
    while not check_termination_conditions(iteration, ITER, start_time, time_limit):

        # Generate all neighbors of the current solution and their moves (the moves are stored in a dictionary)
        # The dictionary is of the form {neighbor: move} where move is a tuple of the form (i, j)
        # neighbors are created by applying the neighbor functions according to function probabilities
        neighbors, moves = generate_neighbors(N_List, N_func_probs, S_current, num_neighbors)

        costs = []
        for neighbor in neighbors:
            costs.append(objective_method(neighbor))

        # sort the neighbors and their costs in ascending order
        # neighbors_and_moves structure: [(cost, neighbor, move), ...]
        # move is a tuple of the form ('function_name', move_tuple)
        neighbors_and_moves: list[tuple[float, list, tuple[str, tuple]]] = sorted(zip(costs, neighbors, moves),
                                                                                  key=lambda x: x[0])

        best_neighbor_found: bool = False
        best_neighbor_cost: float = neighbors_and_moves[0][0]
        best_neighbor_solution: list = neighbors_and_moves[0][1]
        best_neighbor_move: tuple = neighbors_and_moves[0][2]

        index = 0
        while not best_neighbor_found:
            best_neighbor_cost = neighbors_and_moves[index][0]
            best_neighbor_solution = neighbors_and_moves[index][1]
            best_neighbor_move = neighbors_and_moves[index][2]

            meets_aspiration_criteria: bool = False
            in_tabu_list = check_tabu_list(best_neighbor_move,
                                           tabu_list)

            if in_tabu_list:
                meets_aspiration_criteria, tabu_list = check_aspiration_criteria(best_neighbor_move,
                                                                                 best_neighbor_cost,
                                                                                 cost_best,
                                                                                 tabu_list)
                number_of_chased_tabu += 1

            if (not in_tabu_list) or meets_aspiration_criteria:
                best_neighbor_found = True
                if in_tabu_list:
                    number_of_selected_tabu += 1

            index += 1

        # find the index of the best neighbor function
        best_neighbor_move_func_name = best_neighbor_move[0]
        best_neighbor_move_func_index = list(map(lambda x: x.__name__, N_List)).index(best_neighbor_move_func_name)
        N_func_selected[best_neighbor_move_func_index] += 1

        # Update the probabilities of choosing each neighbor function
        if reaction_factor != 1:
            # Update the probabilities of choosing each neighbor function
            N_func_probs = change_function_probabilities_2(func_probs=N_func_probs,
                                                           best_move_func_index=best_neighbor_move_func_index,
                                                           selected_solution_obj=best_neighbor_cost,
                                                           best_solution_obj=cost_best,
                                                           current_solution_obj=cost_current,
                                                           react_factor=reaction_factor)

        # Update the current solution and its cost
        S_current = best_neighbor_solution
        cost_current = best_neighbor_cost

        # Update the gap between the current solution found so far and the benchmark cost
        gap = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
        if gap <= threshold_gap and threshold_gap_time is None and threshold_gap_iter is None:
            threshold_gap_time = time.time() - start_time
            threshold_gap_iter = iteration

        # Add the move that was made to the tabu list
        tabu_list.append(best_neighbor_move)

        # Remove the oldest move from the tabu list if the tabu list is full
        if len(tabu_list) == tabu_size:
            tabu_list.pop(0)

        # Update the iteration counter
        iteration += 1

        # Add the current solution to the list of solutions
        solutions.append(S_current)
        solution_costs.append(cost_current)

        # Update the best solution and its cost if the current solution is better than the best solution found so far
        if cost_current < cost_best:
            S_best = S_current
            cost_best = cost_current
            best_time = time.time() - start_time
            best_iter = iteration

            # track the number of times each neighbor function improved the best solution
            N_func_improves[best_neighbor_move_func_index] += 1

        if print_iteration:
            # print the current iteration information""
            print("\n--------------------------------------------------")
            print("Iteration: ", iteration)
            print("Old solution: ", solutions[-2])
            print("Move: ", best_neighbor_move)
            print("Selected (current) solution: ", S_current)
            print("Current cost: ", cost_current)
            print("Tabu list: ", tabu_list)
            print("Probs of choosing each neighbor function: ", N_func_probs)
            print("\nBest solution: ", S_best)
            print("Best cost: ", cost_best)
            print("Best gap: ", abs(cost_best - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf)
            print("Best time: ", best_time)
            print("Best iteration: ", best_iter)
            print("--------------------------------------------------\n")

        # Check if the algorithm has reached the benchmark optimal cost
        # if benchmark_opt_bool and gap == 0:
        #     break

    time_elapsed = time.time() - start_time  # time elapsed since the start of the algorithm

    if print_results:
        print("-------------------- RESULTS --------------------")
        print("Run parameters:")
        print("Tabu size:", tabu_size)
        print("Number of neighbors:", num_neighbors)
        print("Reaction factor:", reaction_factor)

        print("\nClassic results:")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("Iterations:", iteration)
        print("Time elapsed:", time_elapsed)
        print("Best time:", best_time)
        print("Best iteration:", best_iter)
        print("Initial method:", init_method.__name__)
        print("Initial solution:", S_initial)
        print("Initial cost:", cost_initial)
        print("Benchmark cost:", benchmark_cost)
        print("Threshold gap:", threshold_gap)
        print("Threshold gap time:", threshold_gap_time)
        print("Threshold gap iteration:", threshold_gap_iter)
        print("Time limit:", time_limit)
        print("Iteration limit:", ITER)

        print("\nAlgorithm specific results:")
        print("Number of times a tabu solution was chased and selected:", number_of_chased_tabu)
        print("Number of times a tabu solution met the aspiration criteria and was selected:", number_of_selected_tabu)

        print("\nOther results:")
        print("Number of times each neighbor function was selected:", N_func_selected)
        print("Number of times each neighbor function improved the best solution:", N_func_improves)
        print("Probability of choosing each neighbor function at the end:", N_func_probs)

        print("\n--------------------------------------------------")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("--------------------------------------------------\n")

    if plot_results:
        plt.plot(solution_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Current cost vs Iteration")
        plt.axhline(y=cost_best, color='r', linestyle='-')
        plt.show()

    run_params = {'tabu_size': tabu_size,
                  'num_neighbors': num_neighbors,
                  'reaction_factor': reaction_factor}

    classic_results = {'best_solution': S_best,
                       'best_cost': cost_best,
                       'iterations': iteration,
                       'time_elapsed': time_elapsed,
                       'best_time': best_time,
                       'best_iter': best_iter,
                       'init_method': init_method.__name__,
                       'initial_solution': S_initial,
                       'initial_cost': cost_initial,
                       'benchmark_cost': benchmark_cost,
                       'benchmark_opt_bool': benchmark_opt_bool,
                       'threshold_gap': threshold_gap,
                       'threshold_gap_time': threshold_gap_time,
                       'threshold_gap_iter': threshold_gap_iter,
                       'time_limit': time_limit,
                       'iter_limit': ITER,
                       'date_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    alg_specific_results = {'number_of_chased_tabu': number_of_chased_tabu,
                            'number_of_selected_tabu': number_of_selected_tabu}

    if save_results:
        '''Holder.save_results(algorithm='TS',
                            algorithm_params=run_params,
                            classic_results=classic_results,
                            algorithm_results=alg_specific_results,
                            path=save_results_path,
                            file_name=save_results_file_name)'''
        pass

    return S_best


def SA(init_method: Callable,
       N_List: list[Callable],
       objective_method: Callable,
       init_temp: float = 0.5,
       cooling_rate: float = 0.99,
       N_func_weights: Union[str, list[float | int]] = 'uniform',
       time_limit: float = 1000,
       ITER: int = 1000000,
       benchmark_cost: float = 0,
       benchmark_opt_bool: bool = False,
       threshold_gap: float = 0.01,
       print_iteration: bool = False,
       print_results: bool = False,
       plot_results: bool = False,
       save_results: bool = False,
       save_results_path: str = '',
       save_results_file_name: str = '') -> list:
    """
    Simulated Annealing Algorithm
    """

    # check the parameters
    if len(N_List) == 0:
        # If the list of neighbor functions is empty, raise an error
        msg = "The list of neighbor functions is empty."
        raise ValueError(msg)
    elif len(N_List) != len(N_func_weights) and isinstance(N_func_weights, list):
        # If the number of neighbor functions and the number of probabilities of choosing each neighbor function
        # are not the same, raise an error
        msg = "The number of neighbor functions and the number of probabilities of choosing each neighbor function " \
              "are not the same."
        raise ValueError(msg)

    if isinstance(N_func_weights, str):
        # If the probability distribution of choosing each neighbor function is not a list or 'uniform', raise an error
        if N_func_weights.lower() != 'uniform':
            msg = "The probability distribution of choosing each neighbor function is not valid." \
                  "It should be either 'uniform' or a list of weights."
            raise ValueError(msg)
    elif isinstance(N_func_weights, list):
        # check if all the elements in the list of weights are positive and integers or floats
        for p in N_func_weights:
            if not isinstance(p, (int, float)) or p < 0:
                msg = "The probability distribution of choosing each neighbor function is not valid." \
                      "It should be either 'uniform' or a list of positive weights."
                raise ValueError(msg)

    if init_temp < 0 or init_temp > 1:
        raise ValueError("The initial temperature must be between 0 and 1")

    if cooling_rate < 0 or cooling_rate > 1:
        raise ValueError("The cooling rate must be between 0 and 1")

    if save_results:
        if save_results_path == '':
            msg = "The path to save the results is empty."
            raise ValueError(msg)
        elif save_results_file_name == '':
            msg = "The file name to save the results is empty."
            raise ValueError(msg)

    start_time: float = time.time()  # start time of the algorithm
    non_improvement_time: float = time.time()  # time at which the algorithm stopped improving the best solution

    # Number of neighbor functions
    no_functions: int = len(N_List)

    T_0: float = init_temp  # initial temperature
    alpha: float = cooling_rate  # cooling rate

    T: float = T_0  # current temperature
    T_b: float = T_0  # best temperature
    T_max: float = T_0  # maximum temperature

    # Initialize the solution and its cost
    S_initial: list = init_method()
    # print("INIT DONE")
    cost_initial: float = objective_method(S_initial)

    # Initialize the current solution and its cost
    S_current: list = S_initial
    cost_current: float = cost_initial

    # Initialize the best solution and its cost as the current solution and its cost
    S_best: list = S_current
    cost_best: float = cost_current
    best_time: float = start_time
    best_iter: int = 0

    # Keep track of all the solutions found so far
    solutions: list[list] = [S_current]
    solution_costs: list[float] = [cost_current]

    # Initialize the probabilities of choosing each neighbor function
    N_func_probs: list[float] = [0 for _ in range(no_functions)]
    if isinstance(N_func_weights, str) and N_func_weights.lower() == 'uniform':
        N_func_probs = [1 / no_functions for _ in range(no_functions)]
    elif isinstance(N_func_weights, list):
        N_func_probs = [p / sum(N_func_weights) for p in N_func_weights]

    # Keep track of the gap between the current solution found so far and the benchmark cost
    gap: float = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
    threshold_gap_time: Optional[float] = None
    threshold_gap_iter: Optional[int] = None
    if gap <= threshold_gap:
        threshold_gap_time = time.time() - start_time
        threshold_gap_iter = 0

    Len: int = 0  # number of iterations without improvement of the temperature
    iteration: int = 0  # number of iterations

    # outer loop for the SA algorithm (iteration and time limits)
    for it in range(ITER):

        # Check if the time limit is reached
        if (time.time() - start_time) > time_limit:
            break

        '''# if non improvement time is greater than time limit / 2, stop
        if (time.time() - non_improvement_time) > time_limit / 5:
            break'''

        Len = min(Len + 2, 100)
        # inner loop for the SA algorithm (temperature)
        for i in range(Len):
            iteration += 1

            # Choose a random neighbor function from the list
            rand = random.choices(range(no_functions), weights=N_func_probs)[0]
            n_funk = N_List[rand]

            S_prime = n_funk(S_current)[0]
            cost_prime = objective_method(S_prime)

            if cost_prime < cost_current:
                S_current = S_prime
                cost_current = cost_prime
            else:
                p = math.exp((cost_current - cost_prime) / (T))
                rand1 = random.random()

                if rand1 < p:
                    S_current = S_prime
                    cost_current = cost_prime

            # Check if the current solution is better than the best solution found so far
            if cost_prime < cost_best:
                S_best = S_prime  # update the best solution
                cost_best = cost_prime  # update the best cost
                best_time = time.time() - start_time  # update the best time
                non_improvement_time = time.time()  # if it is not updeted for a long time, stop
                best_iter = iteration  # update the best iteration
                T_b = T  # update the best temperature

            # Update the gap between the current solution found so far and the benchmark cost
            gap = abs(cost_current - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf
            if gap <= threshold_gap and threshold_gap_time is None and threshold_gap_iter is None:
                threshold_gap_time = time.time() - start_time
                threshold_gap_iter = iteration

        T = alpha * T  # decrease the temperature by the cooling rate

        # Check if the temperature is too low
        if T < 0.01:
            T_b = 2 * T_b
            T = min(T_b, T_max)
            Len = 5

        # add the current solution to the list of solutions
        solutions.append(S_current)
        solution_costs.append(cost_current)

        if print_iteration:
            # print the current iteration information""
            print("\n--------------------------------------------------")
            print("Iteration: ", iteration)
            print("Old solution: ", solutions[-2])
            print("Current solution: ", S_current)
            print("Current cost: ", cost_current)
            print("Gap: ", gap)
            print("Temperature: ", T)
            print("Len: ", Len)
            print("\nBest solution: ", S_best)
            print("Best cost: ", cost_best)
            print("Best gap: ", abs(cost_best - benchmark_cost) / benchmark_cost if benchmark_cost != 0 else math.inf)
            print("Best time: ", best_time)
            print("Best iteration: ", best_iter)
            print("Best temperature: ", T_b)
            print("--------------------------------------------------\n")

        # Check if the algorithm has reached the benchmark optimal cost
        # if benchmark_opt_bool and gap == 0:
        #     break

    time_elapsed = time.time() - start_time  # time elapsed since the start of the algorithm

    if print_results:
        print("-------------------- RESULTS --------------------")
        print("Run parameters:")
        print("T_0:", T_0)

        print("\nClassic results:")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("Iterations:", iteration)
        print("Time elapsed:", time_elapsed)
        print("Best time:", best_time)
        print("Best iteration:", best_iter)
        print("Initial method:", init_method.__name__)
        print("Initial solution:", S_initial)
        print("Initial cost:", cost_initial)
        print("Benchmark cost:", benchmark_cost)
        print("Threshold gap:", threshold_gap)
        print("Threshold gap time:", threshold_gap_time)
        print("Threshold gap iteration:", threshold_gap_iter)
        print("Time limit:", time_limit)
        print("Iteration limit:", ITER)

        print("\nAlgorithm specific results:")

        print("\nOther results:")

        print("\n--------------------------------------------------")
        print("Best solution:", S_best)
        print("Best cost:", cost_best)
        print("--------------------------------------------------\n")

    if plot_results:
        plt.plot(solution_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Current cost vs Iteration")
        plt.axhline(y=cost_best, color='r', linestyle='-')
        plt.show()

    run_params = {"T_0": T_0,
                  "alpha": alpha}

    classic_results = {'best_solution': S_best,
                       'best_cost': cost_best,
                       'iterations': iteration,
                       'time_elapsed': time_elapsed,
                       'best_time': best_time,
                       'best_iter': best_iter,
                       'init_method': init_method.__name__,
                       'initial_solution': S_initial,
                       'initial_cost': cost_initial,
                       'benchmark_cost': benchmark_cost,
                       'benchmark_opt_bool': benchmark_opt_bool,
                       'threshold_gap': threshold_gap,
                       'threshold_gap_time': threshold_gap_time,
                       'threshold_gap_iter': threshold_gap_iter,
                       'time_limit': time_limit,
                       'iter_limit': ITER,
                       'date_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    alg_specific_results = {'T_b': T_b,
                            'T_max': T_max,
                            'Len': Len}

    if save_results:
        '''Holder.save_results(algorithm='SA',
                            algorithm_params=run_params,
                            classic_results=classic_results,
                            algorithm_results=alg_specific_results,
                            path=save_results_path,
                            file_name=save_results_file_name)'''
        pass

    return S_best

