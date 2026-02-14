import argparse
from src.solvers import EvolutionaryAlgorithm, PathRepresentation, SelectionMethods
from functools import partial
from Problem import Problem
import networkx as nx

POPULATION_SIZE = 50
OFFSPRING_SIZE = 30

def conversion(best, rep):
    """Convert EA solution (list of trips) to the required flat path format."""
    path = []
    for trip in best.genotype:
        current = 0
        for city, gold in trip:
            sp = rep.sp_path(current, city)
            # add intermediate cities with gold=0
            for node in sp[1:-1]:
                path.append((node, 0))
            path.append((city, gold))
            current = city
        # return to depot
        sp = rep.sp_path(current, 0)
        for node in sp[1:-1]:
            path.append((node, 0))
        path.append((0, 0))
    return path

def is_valid(path, p:Problem):
    # Professor Validation Function 
    for (c1, gold1), (c2, gold2) in zip(path, path[1:]):
        if not nx.has_path(p.graph, c1, c2):
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

    print("Best solution:", last_pop[0].fitness)
    print("baseline", problem.baseline())
    # algorithm.plot(history)

    flatten_path = conversion(last_pop[0], problem_rep)
    print("Is valid path?", is_valid(flatten_path, problem))

    return flatten_path


if __name__ == "__main__":
    solution()
