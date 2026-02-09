import argparse
from src.solvers import EvolutionaryAlgorithm, PathRepresentation, SelectionMethods
from functools import partial
from Problem import Problem

POPULATION_SIZE = 50
OFFSPRING_SIZE = 30

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

    return last_pop[0].genotype + [(0,0)] # to match the expected output format


if __name__ == "__main__":
    solution()
