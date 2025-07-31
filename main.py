import time
import json
from utils.Model import Model
from utils.DataBase import DataBase
from utils.priority_queue import MaxPriorityQueue
from utils.Logger import CustomLogger
from utils.ArgParser import ArgParser
from FeatureSelectorLLM import FeatureSelectorLLM
from utils.SolutionPlot import Plotter

feature_descriptions = {
    "time": "The SV timestamp.",
    "isbA": "Current from Samambaia substation (sb) at Phase A.",
    "isbB": "Current from Samambaia at Phase B.",
    "isbC": "Current from Samambaia at Phase C.",
    "ismA": "Current from Serra da Mesa substation (sm) at Phase A.",
    "ismB": "Current from Serra da Mesa at Phase B.",
    "ismC": "Current from Serra da Mesa at Phase C.",
    "vsbA": "Voltage from Samambaia at Phase A.",
    "vsbB": "Voltage from Samambaia at Phase B.",
    "vsbC": "Voltage from Samambaia at Phase C.",
    "vsmA": "Voltage from Serra da Mesa at Phase A.",
    "vsmB": "Voltage from Serra da Mesa at Phase B.",
    "vsmC": "Voltage from Serra da Mesa at Phase C.",
    "isbARmsValue": "Current RMS Value from Samambaia at Phase A.",
    "isbBRmsValue": "Current RMS Value from Samambaia at Phase B.",
    "isbCRmsValue": "Current RMS Value from Samambaia at Phase C.",
    "ismARmsValue": "Current RMS Value from Serra da Mesa at Phase A.",
    "ismBRmsValue": "Current RMS Value from Serra da Mesa at Phase B.",
    "ismCRmsValue": "Current RMS Value from Serra da Mesa at Phase C.",
    "vsbARmsValue": "Voltage RMS Value from Samambaia at Phase A.",
    "vsbBRmsValue": "Voltage RMS Value from Samambaia at Phase B.",
    "vsbCRmsValue": "Voltage RMS Value from Samambaia at Phase C.",
    "vsmARmsValue": "Voltage RMS Value from Serra da Mesa at Phase A.",
    "vsmBRmsValue": "Voltage RMS Value from Serra da Mesa at Phase B.",
    "vsmCRmsValue": "Voltage RMS Value from Serra da Mesa at Phase C.",
    "isbATrapAreaSum": "Current Trapezoidal Area Sum from Samambaia at Phase A.",
    "isbBTrapAreaSum": "Current Trapezoidal Area Sum from Samambaia at Phase B.",
    "isbCTrapAreaSum": "Current Trapezoidal Area Sum from Samambaia at Phase C.",
    "ismATrapAreaSum": "Current Trapezoidal Area Sum from Serra da Mesa at Phase A.",
    "ismBTrapAreaSum": "Current Trapezoidal Area Sum from Serra da Mesa at Phase B.",
    "ismCTrapAreaSum": "Current Trapezoidal Area Sum from Serra da Mesa at Phase C.",
    "vsbATrapAreaSum": "Voltage Trapezoidal Area Sum from Samambaia at Phase A.",
    "vsbBTrapAreaSum": "Voltage Trapezoidal Area Sum from Samambaia at Phase B.",
    "vsbCTrapAreaSum": "Voltage Trapezoidal Area Sum from Samambaia at Phase C.",
    "vsmATrapAreaSum": "Voltage Trapezoidal Area Sum from Serra da Mesa at Phase A.",
    "vsmBTrapAreaSum": "Voltage Trapezoidal Area Sum from Serra da Mesa at Phase B.",
    "vsmCTrapAreaSum": "Voltage Trapezoidal Area Sum from Serra da Mesa at Phase C.",
    "t": "The timestamp of the last state change.",
    "gooseTimestamp": "The GOOSE timestamp.",
    "sqNum": "The GOOSE sequence number.",
    "stNum": "The GOOSE status number.",
    "cbStatus": "Circuit-breaker status on GOOSE.",
    "frameLen": "The GOOSE ethernet frame length.",
    "ethDst": "The GOOSE ethernet destination address.",
    "ethSrc": "The GOOSE ethernet frame source address.",
    "ethType": "The GOOSE ethernet frame type.",
    "gooseTTL": "The time allowed to live.",
    "gooseAppid": "The GOOSE application ID.",
    "gooseLen": "The GOOSE frame length.",
    "TPID": "The tag priority ID.",
    "gocbRef": "The GOOSE control block reference.",
    "datSet": "The IED dataset path.",
    "goID": "The GOOSE flow ID.",
    "test": "The test flag.",
    "confRev": "The configuration revision.",
    "ndsCom": "The GOOSE NDSCOM parameter.",
    "numDatSetEntries": "The number of entries on the datSet.",
    "APDUSize": "The Application Data Unit (APDU) size.",
    "protocol": "The used protocol (expected: GOOSE).",
    "stDiff": "stNum{n} - stNum{n-1}.",
    "sqDiff": "sqNum{n} - sqNum{n-1}.",
    "gooseLengthDiff": "gooseLen{n} - gooseLen{n-1}.",
    "cbStatusDiff": "cbStatus{n} - cbStatus{n-1}.",
    "apduSizeDiff": "apduSize{n} - apduSize{n-1}.",
    "frameLengthDiff": "frameLen{n} - frameLen{n-1}.",
    "timestampDiff": "time{n} - time{n-1}.",
    "tDiff": "t{n} - t{n-1}.",
    "timeFromLastChange": "gooseTimestamp{n} - t{n}.",
    "delay": "gooseTimestamp{n} - time{n}."
}

def local_search(initial_solution,initial_solution_f1_score, repeated_solutions_count,local_iterations, model: Model,feature_selector: FeatureSelectorLLM, rcl_size,batch_size):
    max_f1_score = initial_solution_f1_score
    best_solution = initial_solution.copy()
    feature_names = model.db.feature_names
    sorted_features = model.db.sorted_features
    logging.info(f"Starting Local Search with initial solution: {initial_solution}, F1-Score: {max_f1_score}")
    new_solutions = []
    new_solutions_scores = []
    seen_solutions = set()
    RCL = [feature for feature, _ in sorted_features[:rcl_size]]
    RCL_indices = [feature_names.index(feature) for feature in RCL]
    local_repetions_count = 0
    invalid_solutions = 0  # Counter for invalid solutions
    for iteration in range(local_iterations):
        new_solution = best_solution.copy()
        new_solution_set =  [feature_names[i] for i in new_solution]
        logging.info(
            f"  → Local Iteration {iteration + 1}/{local_iterations} | Current best F1: {max_f1_score:.4f}")
        if not RCL:
            logging.info("    ✖ RCL is empty. No replacement possible.")
            break
        querry_json = {
            "best_solution": (new_solution,max_f1_score),
            "current_solutions": new_solutions_scores,
            "RCL": RCL_indices
        }
        new_solutions,invalid_solutions_local = feature_selector.select_features(
            f"Your goal is to generate EXACTLY {batch_size} (NEVER MORE THAN THAT) unique feature sets (solutions), each with features that MUST be selected from RCL, based on modifying the best_solution to improve F1-score. "
            f"Do NOT reuse any solution from current_solutions. current_solutions has values from previous iterations and its F1-Score. "
            f"Features MUST be selected from RCL. "
            f"Each solution must be at least the same size as best_solution. "
            f"The input has tuples with the solution and its F1-Score: [solution,F1-score]. "
            f"Inputs in json format: {json.dumps(querry_json)}."
            f"Output MUST be a single VALID JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. Solutions list MUST BE exactly {batch_size} long. This format is required. "
            f"JSON sintax MUST be correct. "
            f"Ensure the solutions are unique and selected feature set was not selected before. "
            f"Do NOT include any explanation, F1-score, text, or Python code..",
            RCL_indices+new_solution
        )

        invalid_solutions += invalid_solutions_local
        new_solutions_scores = []
        for new_solution in new_solutions:
            new_solution_set = frozenset([feature_names[i] for i in new_solution])
            if new_solution_set in seen_solutions:
                repeated_solutions_count += 1
                local_repetions_count += 1
                logging.info(f"    ↺ Duplicate feature combination: {list(new_solution_set)} — Skipping")
            else:
                ### Evaluate the new solution
                sorted_indices = sorted(new_solution)
                f1_score = model.evaluate_algorithm(new_solution)
                logging.info(f"    ✓ Evaluated F1-Score: {f1_score:.4f} for solution: {sorted_indices}")
                new_solutions_scores.append((new_solution, round(f1_score,4)))
                seen_solutions.add(new_solution_set)
                if f1_score > max_f1_score and new_solution_set != frozenset(best_solution):
                    max_f1_score = f1_score
                    best_solution = new_solution
                    sorted_best = sorted(best_solution)
                    logging.info(
                        f"        Improvement found! New best solution: {sorted_best} with F1-Score: {max_f1_score:.4f}")
                elif new_solution_set == frozenset(best_solution):
                    logging.info("No real improvement (same as best solution)")

    logging.info(f"Local Search percent new solutions found: {((batch_size*local_iterations)-local_repetions_count)/(batch_size*local_iterations):.2%}")
    logging.info(f"Local Search completed. Best F1-Score: {max_f1_score}, Best Solution: {best_solution}")
    return max_f1_score, best_solution, repeated_solutions_count, invalid_solutions

def construction(args,model: Model):
    # 'sorted_features' is a list of tuples (feature, IG) sorted by IG. Picking the top X to compose the RCL.
    RCL = [feature for feature, _ in model.db.sorted_features[:args.rcl_size]]
    feature_names = model.db.feature_names
    RCL_indices = [feature_names.index(feature) for feature in RCL]

    RCL_features_dict = {
        feature_names.index(feature): feature_descriptions[feature]
        for feature in RCL
        if feature in feature_descriptions
    }
    logging.info(f"RCL Features: {RCL}")
    logging.info(f"RCL Feature Indices: {RCL_indices}")

    feature_selector = FeatureSelectorLLM(logging)

    all_solutions = []
    local_search_improvements = {}  # Dictionary to store results of local search

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []
    query_json = {
        "RCL": RCL_indices,
    }
    seen_initial_solutions = set()
    repeated_solutions_count = 0  # Initialize the counter for repeated solutions
    repeated_solutions_count_local_search = 0  # Initialize the counter for repeated solutions during local search
    invalid_solutions = 0  # Counter for invalid solutions

    start_time = time.perf_counter()
    if args.rcl_size > len(feature_names):
        raise ValueError("The RCL size cannot exceed the number of available features.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("The initial solution size cannot exceed the RCL size.")

    solutions,invalid_solutions_const = feature_selector.select_features(
        f"Your goal is to generate Exactly {args.constructive_batch} unique feature sets (solutions), which solution must have EXACTLY {args.initial_solution} unique features that MUST be selected from RCL. "
        f"Input in json format: {json.dumps(query_json)}. "
        f"The description of each feature is {RCL_features_dict}. "
        f"Output MUST be a single valid JSON string in following format (with no explanation or code):{{\"solutions\": [[...], [...], ...]}}. This format is required. "
        f"Ensure the solutions are unique (Feature order doesnt make a new solutions. [1,2] and [2,1] are same solution) and selected feature set was not selected before. "
        f"Do NOT include any explanation, text, or Python code.. ",
        RCL_indices
    )
    invalid_solutions += invalid_solutions_const
    iteraction = 0
    for solution in solutions:
        solution_set = frozenset([feature_names[i] for i in solution])
        if solution_set not in seen_initial_solutions:
            seen_initial_solutions.add(solution_set)
            f1_score = model.evaluate_algorithm(solution)
            logging.info(f"F1-Score: {f1_score} for solution: {solution}")
            all_solutions.append((iteraction,f1_score, solution))
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
        else:
            repeated_solutions_count += 1  # Incrementa o contador
            logging.info(f"Repeated initial solution found: {solution}")
        iteraction += 1
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

        improved_f1_score, improved_solution, repeated_solutions_count_local_search,invalid_solutions_local = local_search(
        current_solution,original_f1_score, repeated_solutions_count_local_search,args.local_iterations,model,feature_selector, args.rcl_size,args.local_batch)

        # invalid solutions addition
        invalid_solutions += invalid_solutions_local

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
    logging.info(f"Total invalid solutions encountered: {invalid_solutions}")
    logging.info(f"Initial Solution Size: {solutions}")
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
    logging.info(f"  Constructive Batch: {args.constructive_batch}")
    logging.info(f"  Local Batch: {args.local_batch}")
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

