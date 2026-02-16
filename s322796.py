import argparse
from src.solvers import EvolutionaryAlgorithm, PathRepresentation, SelectionMethods
from functools import partial
from Problem import Problem
import networkx as nx
# This is a basic and fast setup. However, you can adjusted to higher values for better performance.
POPULATION_SIZE = 50
OFFSPRING_SIZE = 30

def conversion(best, rep):
    """Convert EA solution (list of trips) to the required flat path format."""
    path = []
    for trip in best.genotype:
        current = 0
        for city, gold in trip:
            sp = rep.sp_path(current, city)
            for node in sp[1:-1]:
                if node == 0:
                    path.append((0, 0))  # implicit unload
                else:
                    path.append((node, 0))
            path.append((city, gold))
            current = city
        sp = rep.sp_path(current, 0)
        for node in sp[1:-1]:
            if node == 0:
                path.append((0, 0))
            else:
                path.append((node, 0))
        path.append((0, 0))
    return path

def evaluate_path(path, problem):
    """Secondary evaluation function that compute in converted path format.  \n
    This is used to verify the correctness of the conversion and the validity of the path."""
    total_cost = 0.0
    carried = 0.0
    for i in range(len(path) - 1):
        carried += path[i][1]
        c_from, c_to = path[i][0], path[i + 1][0]
        if c_from != c_to:
            total_cost += problem.cost([c_from, c_to], carried)
        if c_to == 0:
            carried = 0.0  # unload at depot
    return total_cost

# This is more complete function that also other people in the group suggest 
def is_valid(problem, path):
    if not path:
        return False
    if not problem.graph.has_edge(0, path[0][0]):
        return False
    
    for (n1, _), (n2, _) in zip(path, path[1:]): # This is match to the part of your validation code !
        if not problem.graph.has_edge(n1, n2):
            return False
    return True

def arg_problem():
    parser = argparse.ArgumentParser(description='Solve the WTCP problem using an evolutionary algorithm.')
    parser.add_argument('--num_cities', type=int, default=100, help='Number of cities in the problem')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for the cost function')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta parameter for the cost function')
    parser.add_argument('--density', type=float, default=1.0, help='Density of the graph (0 to 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    #
    parser.add_argument('--num_generations', type=int, default=50, help='Number of generations for the evolutionary algorithm')
    # This mutation rate yeid mostly better result, and being used in the report. 
    parser.add_argument('--mutation_rate', type=float, default=0.3, help='Mutation rate for the evolutionary algorithm')
    return parser.parse_args()

def solution():

    args = arg_problem()

    problem = Problem(args.num_cities, density=args.density, alpha=args.alpha, beta=args.beta, seed=args.seed)

    problem_rep = PathRepresentation(problem)

    algorithm = EvolutionaryAlgorithm(
        problem_rep=problem_rep,
        population_size=POPULATION_SIZE,
        offspring_size=OFFSPRING_SIZE,
        selection_method=partial(SelectionMethods.tournament_selection, tournament_size=10)
    )

    last_pop, history = algorithm.solve(num_generations=args.num_generations, mutation_rate=args.mutation_rate)

    # algorithm.plot(history)

    flatten_path = conversion(last_pop[0], problem_rep)
    print("Is valid path?", is_valid(problem, flatten_path))

    # use uncomment to print the raw fitness from the EA, 
    # which it should be equal to the evaluated cost from the converted path !!!
    # print("Best solution:", last_pop[0].fitness)
    print("Evaluated Best path cost:", evaluate_path(flatten_path, problem))
    print("baseline", problem.baseline())

    return flatten_path


if __name__ == "__main__":
    solution()
