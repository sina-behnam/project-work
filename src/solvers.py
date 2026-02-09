import numpy as np
import random
from collections import namedtuple
from tqdm import tqdm
from collections import deque, defaultdict
from itertools import permutations
from Problem import Problem
import networkx as nx

Individual = namedtuple('Individual', ['genotype', 'fitness'])

class PathRepresentation:

    def __init__(self, problem: Problem):
        self.problem = problem
        self._G = problem.graph  # store one copy
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.cities = [n for n in self._G.nodes if n != 0]
        self.golds = {n: self._G.nodes[n]['gold'] for n in self.cities}
        self._pos = nx.get_node_attributes(self._G, 'pos')
        self._n = len(self._G.nodes)
        self._sp_cache = {}
        self._baseline_cost = None

        # Euclidean matrix
        coords = np.array([self._pos[i] for i in range(self._n)])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self._euc = np.sqrt(np.sum(diff**2, axis=-1))

        # check density ( it could have been provided by the problem but we can re-compute it here too )
        actual_edges = len(self._G.edges)
        possible_edges = self._n * (self._n - 1) / 2
        self._use_direct = (actual_edges / possible_edges) > 0.99 if possible_edges > 0 else False

    def euc_dist(self, a, b):
        return self._euc[a, b]

    def sp_dist(self, a, b):
        if a == b:
            return 0.0
        if self._use_direct:
            return self._euc[a, b]
        if (a, b) not in self._sp_cache:
            self._compute_and_cache(a, b)
        return self._sp_cache[(a, b)][0]

    def sp_path(self, a, b):
        if a == b:
            return [a]
        if self._use_direct:
            return [a, b]
        if (a, b) not in self._sp_cache:
            self._compute_and_cache(a, b)
        return self._sp_cache[(a, b)][1]

    def _compute_and_cache(self, a, b):
        path = nx.astar_path(self._G, a, b,
                             heuristic=lambda u, v: self._euc[u, v],
                             weight='dist')
        dist = nx.path_weight(self._G, path, weight='dist')
        self._sp_cache[(a, b)] = (dist, path)
        self._sp_cache[(b, a)] = (dist, list(reversed(path)))

    def cost(self, path, weight):
        return self.problem.cost(path, weight)

    def cached_baseline(self):
        if self._baseline_cost is None:
            self._baseline_cost = self.problem.baseline()
        return self._baseline_cost

    def eval_trip(self, trip):
        c, w = 0, 0.0
        cost = 0.0
        for city, g in trip:
            if c != city:
                path = self.sp_path(c, city)
                for a, b in zip(path, path[1:]):
                    cost += self.cost([a, b], w)
            w += g
            c = city
        if c != 0:
            path = self.sp_path(c, 0)
            for a, b in zip(path, path[1:]):
                cost += self.cost([a, b], w)
        return cost

    def eval_solution(self, sol):
        return sum(self.eval_trip(t) for t in sol)
    
    def optimize_trip_order(self, trip):
        """For beta >= 1 we want heavy segments short then try all permutations
        for small trips, greedy for larger ones."""
        if len(trip) <= 1:
            return trip
        if len(trip) <= 6:
            best, best_c = trip, self.eval_trip(trip)
            for perm in permutations(trip):
                c = self.eval_trip(list(perm))
                if c < best_c:
                    best, best_c = list(perm), c
            return best
        # greedy: pick next city that minimizes incremental cost
        remaining = list(trip)
        ordered = []
        cur, w = 0, 0.0
        while remaining:
            best_i, best_c = 0, float('inf')
            for i, (city, g) in enumerate(remaining):
                c = sum(self.cost([a, b], w) for a, b in
                        zip(self.sp_path(cur, city), self.sp_path(cur, city)[1:]))
                if c < best_c:
                    best_i, best_c = i, c
            city, g = remaining.pop(best_i)
            ordered.append((city, g))
            w += g
            cur = city
        return ordered

class SelectionMethods:

    @staticmethod
    def roulette_wheel(population : list[Individual]) -> Individual:
        # 
        max_fitness = max(ind.fitness for ind in population)
        weights = [(max_fitness - ind.fitness + 1) for ind in population]  # +1 to avoid zero
        return random.choices(population, weights=weights, k=1)[0]
    
    @staticmethod
    def tournament_selection(population : list[Individual], tournament_size=3) -> Individual:
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda ind: ind.fitness, reverse=False)
        return tournament[0]

class MutationOperators:

    @staticmethod
    def swap(sol: Individual, rep: PathRepresentation) -> Individual:
        """Swap 2 cases, swap cities in a trip, swap cities between trips and merging them if needed."""
        trips = [list(t) for t in sol.genotype]
        flat = [(i, j) for i, t in enumerate(trips) for j in range(len(t))]
        if len(flat) < 2:
            return sol
        (i1, j1), (i2, j2) = random.sample(flat, 2)
        if i1 == i2:
            # same trip — safe to swap, no duplicates
            trips[i1][j1], trips[i1][j2] = trips[i1][j2], trips[i1][j1]
        else:
            c1, g1 = trips[i1][j1]
            c2, g2 = trips[i2][j2]
            trip1_cities = {c for c, g in trips[i1]} - {c1}
            trip2_cities = {c for c, g in trips[i2]} - {c2}

            if c2 in trip1_cities:
                # c2 already in trip1 — merge gold
                trips[i1] = [(c, g + g2) if c == c2 else (c, g)
                             for c, g in trips[i1] if (c, g) != (c1, g1)]
            else:
                trips[i1][j1] = (c2, g2)

            if c1 in trip2_cities:
                trips[i2] = [(c, g + g1) if c == c1 else (c, g)
                             for c, g in trips[i2] if (c, g) != (c2, g2)]
            else:
                trips[i2][j2] = (c1, g1)

        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def change_k(sol: Individual, rep: PathRepresentation) -> Individual:
        """Change number of trips for a random city (beta>1)."""
        trips = [list(t) for t in sol.genotype]
        city = random.choice(rep.cities)
        idxs_golds = [(i, g) for i, t in enumerate(trips) for c, g in t if c == city]
        if not idxs_golds:
            return sol
        total_g = sum(g for _, g in idxs_golds)
        for i, _ in sorted(idxs_golds, key=lambda x: x[0], reverse=True):
            trips[i] = [(c, g) for c, g in trips[i] if c != city]
        trips = [t for t in trips if t]
        old_k = len(idxs_golds)
        new_k = max(1, old_k + random.choice([-2, -1, 1, 2]))
        gk = total_g / new_k
        for _ in range(new_k):
            trips.append([(city, gk)])
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def merge_trips(sol: Individual, rep: PathRepresentation) -> Individual:
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 2:
            return sol
        i, j = random.sample(range(len(trips)), 2)

        # merge with duplicate handling
        merged = list(trips[i])
        merged_cities = {c for c, g in merged}
        for city, gold in trips[j]:
            if city in merged_cities:
                merged = [(c, g + gold) if c == city else (c, g)
                          for c, g in merged]
            else:
                merged.append((city, gold))
                merged_cities.add(city)

        merged = rep.optimize_trip_order(merged) # This function before was used for initialization too !
        trips = [t for k, t in enumerate(trips) if k not in (i, j)]
        trips.append(merged)
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def split_trip(sol: Individual, rep: PathRepresentation) -> Individual:
        """Split a trip at a random point."""
        trips = [list(t) for t in sol.genotype]
        idx = random.randint(0, len(trips) - 1)
        if len(trips[idx]) < 2:
            return sol
        mid = random.randint(1, len(trips[idx]) - 1)
        a, b = trips[idx][:mid], trips[idx][mid:]
        trips[idx] = a
        trips.append(b)
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def move_city(sol: Individual, rep: PathRepresentation) -> Individual:
        """Move a stop from one trip to another, merging gold if city already exists."""
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 2:
            return sol

        src = random.randint(0, len(trips) - 1)
        if not trips[src]:
            return sol

        item = trips[src].pop(random.randint(0, len(trips[src]) - 1))
        city, gold = item

        dst = random.randint(0, len(trips) - 1)
        dst_cities = {c for c, g in trips[dst]}

        if city not in dst_cities:
            # insert at random position
            trips[dst].insert(random.randint(0, len(trips[dst])), item)
        else:
            # merge gold into existing entry
            trips[dst] = [(c, g + gold) if c == city else (c, g)
                          for c, g in trips[dst]]

        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

class CrossoverOperators:

    @staticmethod
    def trip_crossover(p1: list, p2: list, rep: PathRepresentation) -> Individual:
        k = random.randint(1, max(1, len(p1) // 2))
        child_trips = [list(t) for t in random.sample(p1, k)]

        covered = defaultdict(float)
        for t in child_trips:
            for c, g in t:
                covered[c] += g

        for t in p2:
            new_t = []
            for c, g in t:
                need = rep.golds[c] - covered[c]
                if need > 0.01:
                    take = min(g, need)
                    # check if city already in new_t
                    existing = [i for i, (cc, gg) in enumerate(new_t) if cc == c]
                    if existing:
                        idx = existing[0]
                        new_t[idx] = (c, new_t[idx][1] + take)
                    else:
                        new_t.append((c, take))
                    covered[c] += take
            if new_t:
                child_trips.append(new_t)

        for c in rep.cities:
            need = rep.golds[c] - covered.get(c, 0)
            if need > 0.01:
                child_trips.append([(c, need)])

        return Individual(child_trips, rep.eval_solution(child_trips))

    @staticmethod
    def targeted_crossover(p1: Individual, p2: Individual, rep: PathRepresentation) -> Individual:
        trips1 = p1.genotype
        trips2 = p2.genotype

        group1 = {}
        for t in trips1:
            cities_in_trip = frozenset(c for c, g in t)
            for c, _ in t:
                group1[c] = cities_in_trip

        group2 = {}
        for t in trips2:
            cities_in_trip = frozenset(c for c, g in t)
            for c, _ in t:
                group2[c] = cities_in_trip

        shared_pairs = set()
        for c in rep.cities:
            if c in group1 and c in group2:
                common = group1[c] & group2[c]
                if len(common) > 1:
                    shared_pairs.add(frozenset(common))

        child_trips = [list(t) for t in trips1]

        for shared_group in shared_pairs:
            if len(shared_group) < 2 or random.random() > 0.5:
                continue

            target_idx = None
            for i, t in enumerate(child_trips):
                trip_cities = {c for c, g in t}
                if shared_group.issubset(trip_cities):
                    target_idx = i
                    break

            if target_idx is None:
                continue

            trip = child_trips[target_idx]
            shared_list = [item for item in trip if item[0] in shared_group]
            n_pull = random.randint(1, max(1, len(shared_list) // 2))
            to_pull = set(c for c, g in random.sample(shared_list, n_pull))

            stay = [(c, g) for c, g in trip if c not in to_pull]
            pulled = [(c, g) for c, g in trip if c in to_pull]

            child_trips[target_idx] = stay

            if random.random() < 0.5 or len(child_trips) < 2:
                child_trips.append(pulled)
            else:
                merge_idx = random.randint(0, len(child_trips) - 1)
                # merge with duplicate check
                existing_cities = {c for c, g in child_trips[merge_idx]}
                for city, gold in pulled:
                    if city in existing_cities:
                        child_trips[merge_idx] = [
                            (c, g + gold) if c == city else (c, g)
                            for c, g in child_trips[merge_idx]
                        ]
                    else:
                        child_trips[merge_idx].append((city, gold))
                child_trips[merge_idx] = rep.optimize_trip_order(child_trips[merge_idx])

        child_trips = [t for t in child_trips if t]
        return Individual(child_trips, rep.eval_solution(child_trips))
    
class EvolutionaryAlgorithm:

    def __init__(self,
                   problem_rep : PathRepresentation,
                   population_size=100,
                   offspring_size=50,
                   selection_method : callable = None
                   ):

        self.representation = problem_rep
        
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.selection_method = selection_method

        # Initialize the current population
        self.current_population = self.initialize_population()

    def initialize_genome(self):
        sol = []
        for c in self.representation.cities:
            g, d = self.representation.golds[c], self.representation.euc_dist(0, c)
            best_k, best_cost = 1, float('inf')
            for k in range(1, 50):
                gk = g / k
                per = 2 * d + (self.representation.alpha * d * gk) ** self.representation.beta
                total = k * per
                if total < best_cost:
                    best_k, best_cost = k, total
                elif total > best_cost * 1.05:
                    break
            gk = g / best_k
            for _ in range(best_k):
                sol.append([(c, gk)])
        return sol

    def initialize_population(self) -> list[Individual]:
        population = []
        for i in tqdm(range(self.population_size), desc="Initializing population"):
            genotype = self.initialize_genome()
            fitness = self.representation.eval_solution(genotype)
            population.append(Individual(genotype, fitness))
        population.sort(key=lambda ind: ind.fitness)
        return population

                
    def evaluate_population(self, offspring: list[Individual]) -> list[Individual]:
        # (mu + lambda) selection: merge parents + offspring, keep best
        combined = self.current_population + offspring
        combined.sort(key=lambda ind: ind.fitness)
        return combined[:self.population_size]

    def solve(self, num_generations=100, mutation_rate=0.3):
        history = []

        if self.representation.beta > 1:
            mut_ops = [
                (MutationOperators.change_k, 0.5),
                (MutationOperators.split_trip, 0.2),
                (MutationOperators.swap, 0.3),
            ]
        else:
            mut_ops = [
                (MutationOperators.merge_trips, 0.4),
                (MutationOperators.move_city, 0.4),
                (MutationOperators.swap, 0.2)
            ]

        ops, weights = zip(*mut_ops)

        for generation in tqdm(range(num_generations), desc="Evolving"):
            offspring = []

            for _ in range(self.offspring_size):
                parent1 = self.selection_method(self.current_population)

                if random.random() > mutation_rate:
                    parent2 = self.selection_method(self.current_population)
                    
                    if self.representation.beta > 1:
                        child = CrossoverOperators.trip_crossover(
                            parent1.genotype, parent2.genotype, self.representation)
                    else:
                        child = CrossoverOperators.targeted_crossover(parent1, parent2, self.representation)
                else:
                    mut_fn = random.choices(ops, weights=weights, k=1)[0]
                    child = mut_fn(parent1, self.representation)

                offspring.append(child)

            self.current_population = self.evaluate_population(offspring)
            history.append(self.current_population[0].fitness)

        return self.current_population, history


    def validate_solution(self, sol):
        """Check that total gold in solution matches total gold in graph."""
        sol_gold = defaultdict(float)
        for trip in sol:
            for city, gold in trip:
                sol_gold[city] += gold
    
        valid = True
        for city in self.representation.cities:
            expected = self.representation.golds[city]
            actual = sol_gold.get(city, 0)
            if abs(expected - actual) > 0.01:
                print(f"City {city}: expected {expected:.2f}, got {actual:.2f}")
                valid = False
    
        # check no extra cities
        for city in sol_gold:
            if city not in self.representation.cities:
                print(f"City {city}: not in graph but found in solution")
                valid = False
    
        total_expected = sum(self.representation.golds.values())
        total_actual = sum(sol_gold.values())
        print(f"Total gold — expected: {total_expected:.2f}, actual: {total_actual:.2f}, "
              f"match: {abs(total_expected - total_actual) < 0.01}")
    
        return valid

    def plot(self, history):
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Evolution of Best Fitness Over Generations')
        plt.grid()
        plt.show()
