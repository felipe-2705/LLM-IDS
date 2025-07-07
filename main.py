import time
from utils.Model import Model
from utils.DataBase import DataBase
from utils.priority_queue import MaxPriorityQueue
from utils.Logger import CustomLogger
from utils.ArgParser import ArgParser
from FeatureSelectorLLM import FeatureSelectorLLM
from utils.SolutionPlot import Plotter

def local_search(initial_solution, repeated_solutions_count,local_iterations, model: Model,feature_selector: FeatureSelectorLLM, rcl_size):
    max_f1_score = model.evaluate_algorithm(initial_solution)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}
    feature_names = model.db.feature_names
    sorted_features = model.db.sorted_features

    logging.info(f"Starting Local Search with initial solution: {initial_solution}, F1-Score: {max_f1_score}")

    for iteration in range(local_iterations):
        new_solution = best_solution.copy()

        logging.info(
            f"  → Local Iteration {iteration + 1}/{local_iterations} | Current best F1: {max_f1_score:.4f}")

        tries = len(new_solution)
        while tries>0:
            RCL = [feature_names.index(feature) for feature, score in sorted_features[:rcl_size]
                   if feature_names.index(feature) not in new_solution]
            if not RCL:
                logging.info("    ✖ RCL is empty. No replacement possible.")
                break

            new_selected_features = feature_selector.select_features(
                f"choose 1 feature from the current solution {best_solution} to replace"
                f"Replace de choose feature with 1 feature from the following list: {', '.join(str(idx) for idx in RCL)}, that is not in the current solution."
                f"The response should be a new solution with some form as current solution."
                f"The list containes features with high Mutual Information scores."
                f"Ensure the selected features are unique and relevant for the task at hand."
            )
            new_solution = [feature_names.index(feature_name) for feature_name in new_selected_features]

            new_solution_set = frozenset(new_solution)
            if new_solution_set in seen_solutions:
                repeated_solutions_count += 1
                logging.info(f"    ↺ Duplicate feature combination: {list(new_solution_set)} — Skipping")
                tries -= 1
                continue
            else:
                break

        sorted_indices = sorted(new_solution)
        f1_score = model.evaluate_algorithm(new_solution)
        logging.info(f"    ✓ Evaluated F1-Score: {f1_score:.4f} for solution: {sorted_indices}")

        if f1_score > max_f1_score and new_solution_set != frozenset(best_solution):
            max_f1_score = f1_score
            best_solution = new_solution
            seen_solutions.add(new_solution_set)
            sorted_best = sorted(best_solution)
            logging.info(
                f"        Improvement found! New best solution: {sorted_best} with F1-Score: {max_f1_score:.4f}")
        elif new_solution_set == frozenset(best_solution):
            logging.info("No real improvement (same as best solution)")

    logging.info(f"Local Search completed. Best F1-Score: {max_f1_score}, Best Solution: {best_solution}")
    return max_f1_score, best_solution, repeated_solutions_count

def construction(args,model: Model):
    # 'sorted_features' is a list of tuples (feature, IG) sorted by IG. Picking the top X to compose the RCL.
    RCL = [feature for feature, _ in model.db.sorted_features[:args.rcl_size]]
    feature_names = model.db.feature_names
    RCL_indices = [feature_names.index(feature) for feature in RCL]

    logging.info(f"RCL Features: {RCL}")
    logging.info(f"RCL Feature Indices: {RCL_indices}")

    feature_selector = FeatureSelectorLLM(logging)

    all_solutions = []
    local_search_improvements = {}  # Dictionary to store results of local search

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []

    seen_initial_solutions = set()
    repeated_solutions_count = 0  # Initialize the counter for repeated solutions
    repeated_solutions_count_local_search = 0  # Initialize the counter for repeated solutions during local search

    start_time = time.perf_counter()
    if args.rcl_size > len(feature_names):
        raise ValueError("The RCL size cannot exceed the number of available features.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("The initial solution size cannot exceed the RCL size.")

    for iteration in range(args.constructive_iterations):
        # Ensure the initial solution is unique
        while True:
            # Randomly select k features from RCL to generate initial solutions
           ## selected_features = random.sample(RCL, k=args.initial_solution)
            selected_features = feature_selector.select_features(
                f"Select {args.initial_solution} features from the following list: {', '.join(str(idx) for idx in RCL)}. "
                f"The list containes features with high Mutual Information scores."
                f"Ensure the selected features are unique and relevant for the task at hand."
            )
            # Convert feature names into indices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                logging.info(f"Repeated initial solution found: {solution}, generating a new solution...")

        f1_score = model.evaluate_algorithm(solution)
        logging.info(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.0:
            # If the priority queue is not full, simply insert the new F1-Score.
            if len(priority_queue.heap) < args.priority_queue:
                priority_queue.insert((f1_score, solution))
            else:
                # If the priority queue is full, find the lowest F1-Score in the queue.
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove the item with the lowest F1-Score before inserting the new item.
                    priority_queue.heap.remove((lowest_f1, [item[1] for item in priority_queue.heap if item[0] == lowest_f1][0]))
                    priority_queue.insert((f1_score, solution))
        local_search_improvements[tuple(solution)] = 0

    # visualize_heap(priority_queue.heap)
    total_elapsed_time = time.perf_counter() - start_time
    logging.info(f"Total repeated initial solutions: {repeated_solutions_count}")
    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    print_priority_queue(priority_queue)
    plotter = Plotter(priority_queue, logging)
    plotter.plot_solutions_with_priority(all_solutions)

    start_time = time.perf_counter()  # Local Search Phase
    total_iterations = len(priority_queue.heap) * args.local_iterations  # Total predicted iterations
    queue_progress = 0

    while not priority_queue.is_empty():
        original_f1_score , current_solution = priority_queue.extract_max()

        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(
        current_solution, repeated_solutions_count_local_search,args.local_interations, args.algorithm, args.rcl_size)

        # Increment iteration count
        queue_progress += 1

        # Progress log
        elapsed_time = time.perf_counter() - start_time
        estimated_total_time = (elapsed_time / queue_progress) * total_iterations
        logging.info(
            f"[{queue_progress}/{args.priority_queue}] Best solution: F1-Score {improved_f1_score:.4f} |"
            f" Estimated remaining time: {estimated_total_time - elapsed_time:.2f}s")

        # Check if there was an improvement compared to the original F1-Score of the specific solution
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            logging.info(f"Improvement in Local Search! F1-Score: {improved_f1_score} for solution: {current_solution}. New solution: {improved_solution}")

        # Check if the improved solution is the global best solution
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            logging.info(f"New Global Best Solution! F1-Score: {max_f1_score} for solution: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local

    plotter.plot_solutions(all_solutions, local_search_improvements)

    logging.info(f"Total repeated solutions in local search: {repeated_solutions_count_local_search}")
    logging.info(f"Initial Solution Size: {selected_features}")
    logging.info(f"RCL Size: {len(RCL)}")
    logging.info(f"Best F1-Score: {max_f1_score}")
    logging.info(f"Best Feature Set (indices): {best_solution}")

    # Map indices to feature names
    best_feature_names = [(feature_names[i], i) for i in best_solution]
    formatted_best_features = ", ".join([f"'{name}' ({index})" for name, index in best_feature_names])

    logging.info(f"Best Feature Set (names): {formatted_best_features}")

    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    logging.info(f"Total execution time for Local Search Phase: {total_local_search_time} seconds")

def print_priority_queue(priority_queue):
    logging.info("Priority Queue:")
    for score, solution in priority_queue.heap:
        logging.info(f"F1-Score: {-score}, Solution: {solution}")

if __name__ == '__main__':
    args = ArgParser().get_args()
    # Initialize the custom       
    logging = CustomLogger(debug=args.debug).logger
    logging.info("Execution parameters:")
    logging.info(f"  Algorithm: {args.algorithm}")
    logging.info(f"  RCL Size: {args.rcl_size}")
    logging.info(f"  Initial Solution Size: {args.initial_solution}")
    logging.info(f"  Priority Queue Size: {args.priority_queue}")
    logging.info(f"  Local Search Iterations: {args.local_iterations}")
    logging.info(f"  Constructive Iterations: {args.constructive_iterations}")
    logging.info(f"  Debug Mode: {'Enabled' if args.debug else 'Disabled'}")    
    logging.info("-" * 50)


    # Load and preprocess the data
    db = DataBase(logging) 
    db.load_and_preprocess()
    model =  Model(algorithm=args.algorithm,database=db,logger=logging)

    # Print IG scores
    db.rank_features()
    db.print_feature_scores()

    # Initial evaluation (baseline)
    baseline_f1 = model.evaluate_baseline()

    # Continue with the selected algorithm for the next steps
    logging.info(f"Selected algorithm for constructive and local search phases: {args.algorithm.upper()}")

    # Execute construction and local search
    construction(args,model)
    logging.info(f"Baseline F1-Score (All Features with {args.algorithm.upper()}): {baseline_f1:.4f}")

