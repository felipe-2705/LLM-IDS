import argparse

class ArgParser():
    """
    Class to handle command line arguments for the LLM-FS algorithm.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
                description="Use LLM For feature Selection.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        self.parser.add_argument(
            "-a", "--algorithm", "--alg",
            type=str, choices=['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost', 'linear_svc', 'sgd'],
            default='nb',
            help="Algorithm to be used for evaluation (knn, dt, nb, rf, svm, xgboost, linear_svc, sgd)."
        )
        self.parser.add_argument(
            "-rcl", "--rcl_size", "--rcl",
            type=int, default=10,
            help="Size of the Restricted Candidate List (RCL)."
        )
        self.parser.add_argument(
            "-is", "--initial_solution", "--init_sol",
            type=int, default=5,
            help="Size of the initial solution generated."
        )
        self.parser.add_argument(
            "-pq", "--priority_queue", "--pq_size",
            type=int, default=10,
            help="Maximum size of the priority queue."
        )
        self.parser.add_argument(
            "-lc", "--local_iterations", "--ls",
            type=int, default=50,
            help="Number of iterations in the local search phase."
        )
        self.parser.add_argument(
            "-cc", "--constructive_iterations", "--const",
            type=int, default=100,
            help="Number of iterations in the constructive phase."
        )
        self.parser.add_argument(
            "-d", "--debug",
            action="store_true",
            help="Enable debug mode."
        )
        self.args = self.parser.parse_known_args()

    def get_args(self):
        return self.args

    def __str__(self):
        return f"Arguments: {self.args}"
    
    def __repr__(self):
        return f"ArgParser({self.args})"