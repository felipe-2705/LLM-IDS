from matplotlib import pyplot as plt
from utils.Logger import CustomLogger
from utils.priority_queue import MaxPriorityQueue

class Plotter: 
    def __init__(self, priority_queue: MaxPriorityQueue,  logger: CustomLogger):
        """
        Initialize the Plotter with an optional logger.
        :param logger: A CustomLogger object for logging operations (optional).
        """
        self.logger = logger
        self.priority_queue = priority_queue
        self.results_path = "results"

    def plot_solutions_with_priority(self,all_solutions):
        # Convert the priority queue into a set for fast lookup
        self.logger.debug("Converting priority queue to set for fast lookup.")
        priority_set = set([tuple(sol) for _, sol in self.priority_queue.heap])

        # Extracting iteration indices and F1-Scores
        self.logger.debug("Extracting iteration indices and F1-Scores from all solutions.")
        iterations = [iteration for _, iteration, _ in all_solutions]
        f1_scores = [f1 for f1, _, _ in all_solutions]

        # Checking which solutions are in the top 10
        self.logger.debug("Determining priority colors for solutions.")
        priority_colors = ['red' if tuple(sol) in priority_set else 'blue' for _, _, sol in all_solutions]

        plt.scatter(f1_scores, iterations, color=priority_colors)
        plt.ylabel('F1-Score')
        plt.xlabel('Índice da Solução')
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Top 10', markersize=10, markerfacecolor='red'),
                        plt.Line2D([0], [0], marker='o', color='w', label='Outras Soluções', markersize=10, markerfacecolor='blue')],
                loc='lower right')
        plt.savefig(f"{self.results_path}/priority_plot.png")


    def plot_solutions(self,all_solutions, local_search_improvements):
        plt.figure(figsize=(9, 4))
        priority_queue_snapshot= list(self.priority_queue.heap)
        # Convert the priority queue into a set for fast lookup
        self.logger.debug("Converting priority queue to set for fast lookup.")
        priority_set = set([tuple(sol) for _, sol in priority_queue_snapshot])

        # Extracting iteration indices and F1-Scores
        self.logger.debug("Extracting iteration indices and F1-Scores from all solutions.")
        iterations = [iteration for iteration, _, _ in all_solutions]
        f1_scores = [f1 for _, f1, _ in all_solutions]
        solutions = [sol for _, _, sol in all_solutions]

        # Draw all blue bars first
        plt.bar(iterations, f1_scores, color='blue', label='Soluções Iniciais (SI)')

        # Draw bars for priority queue solutions in red
        self.logger.debug("Highlighting solutions in the priority queue.")
        for i, sol in enumerate(solutions):
            if tuple(sol) in priority_set:
                plt.bar(iterations[i], f1_scores[i], color='red', label='SI Incluídas na Fila de Prioridades')

        # Overpaint improvements in green where applicable
        self.logger.debug("Highlighting improvements from local search.")
        for i, sol in enumerate(solutions):
            improvement = local_search_improvements.get(tuple(sol), 0)
            if improvement > 0:
                plt.bar(iterations[i], improvement, bottom=f1_scores[i], color='green', label='SI Melhoradas na Busca Local')

        self.logger.debug("Setting plot labels and legend.")
        plt.xlabel('Índice da Solução', fontsize=12, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # by_label = {label: handle for label, handle in by_label.items() if label in ['Soluções Iniciais', 'Top 10', 'Melhoradas na Busca Local']}
        plt.legend(by_label.values(), by_label.keys(), loc='lower right', prop={'weight': 'bold'})
        plt.xlim(min(iterations) - 1, max(iterations) + 1)
        plt.tick_params(axis='x', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo x
        plt.tick_params(axis='y', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo y
        plt.tight_layout()

        plt.savefig(f"{self.results_path}/all_bestsolution.png")
        plt.savefig(f"{self.results_path}/all_bestsolution.pdf")
