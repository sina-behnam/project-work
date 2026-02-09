import numpy as np
import random
from collections import namedtuple
from tqdm import tqdm
from collections import deque, defaultdict
from itertools import permutations
from problem import Problem
import networkx as nx

Individual = namedtuple('Individual', ['genotype', 'fitness'])


class PathRepresentation(Problem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        G = self.graph
        self.cities = [n for n in G.nodes if n != 0]
        self.golds = {n: G.nodes[n]['gold'] for n in self.cities}

        # precompute all shortest paths
        print("Precomputing shortest paths between all pairs of nodes...")
        if 'density' in kwargs and kwargs['density'] > 0.7 and len(G.nodes) > 500:
            print('IT may take longer time')
        
        self._sp = dict(nx.all_pairs_dijkstra(G, weight='dist'))
        self._baseline_cost = None

    def sp_dist(self, a, b):
        return self._sp[a][0][b]

    def sp_path(self, a, b):
        return self._sp[a][1][b]

    def cached_baseline(self):
        if self._baseline_cost is None:
            self._baseline_cost = self.baseline()
        return self._baseline_cost

    def eval_trip(self, trip):
        """trip = [(city, gold_amount), ...] then cost including return"""
        c, w = 0, 0.0
        cost = 0.0
        for city, g in trip:
            for a, b in zip(self.sp_path(c, city), self.sp_path(c, city)[1:]):
                cost += self.cost([a, b], w)
            w += g
            c = city
        for a, b in zip(self.sp_path(c, 0), self.sp_path(c, 0)[1:]):
            cost += self.cost([a, b], w)
        return cost

    def eval_solution(self, sol):
        return sum(self.eval_trip(t) for t in sol)
    
    def optimize_trip_order(self, trip):
        """For beta >= 1 we want heavy segments short then try all permutations
        for small trips, greedy for larger ones."""
        if len(trip) <= 1:
            return trip
        if len(trip) <= 7:
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
        """Swap two stops across trips."""
        trips = [list(t) for t in sol.genotype]
        flat = [(i, j) for i, t in enumerate(trips) for j in range(len(t))]
        if len(flat) < 2:
            return sol
        (i1, j1), (i2, j2) = random.sample(flat, 2)
        trips[i1][j1], trips[i2][j2] = trips[i2][j2], trips[i1][j1]
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
        """Merge two trips into one (beta=1)."""
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 2:
            return sol
        i, j = random.sample(range(len(trips)), 2)
        merged = trips[i] + trips[j]
        merged = rep.optimize_trip_order(merged)
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
        """Move a stop from one trip to another."""
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 2:
            return sol
        src = random.randint(0, len(trips) - 1)
        if not trips[src]:
            return sol
        item = trips[src].pop(random.randint(0, len(trips[src]) - 1))
        dst = random.randint(0, len(trips) - 1)
        trips[dst].insert(random.randint(0, len(trips[dst])), item)
        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

class AStarGuidedOperators:

    @staticmethod
    def insertion_mutate(sol: Individual, rep: PathRepresentation) -> Individual:
        """Remove a city from one trip, insert it at the cheapest 
        position in any trip using distance heuristic."""
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 1:
            return sol

        # pick random city to relocate
        src_idx = random.randint(0, len(trips) - 1)
        if not trips[src_idx]:
            return sol
        pos = random.randint(0, len(trips[src_idx]) - 1)
        item = trips[src_idx].pop(pos)
        city, g = item

        # score each possible insertion using A* distances
        best_trip, best_pos, best_score = None, None, float('inf')

        for t_idx, trip in enumerate(trips):
            for ins_pos in range(len(trip) + 1):
                # cheaply estimate cost of inserting here
                prev = trip[ins_pos - 1][0] if ins_pos > 0 else 0  # depot
                nxt = trip[ins_pos][0] if ins_pos < len(trip) else 0  # depot

                # cost of detour: prev→city→nxt instead of prev→nxt
                d_detour = rep.sp_dist(prev, city) + rep.sp_dist(city, nxt)
                d_direct = rep.sp_dist(prev, nxt)
                score = d_detour - d_direct  # lower = better fit

                if score < best_score:
                    best_trip, best_pos, best_score = t_idx, ins_pos, score

        trips[best_trip].insert(best_pos, item)
        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def swap_guided(sol: Individual, rep: PathRepresentation) -> Individual:
        """Swap two cities between trips if it reduces total detour cost."""
        trips = [list(t) for t in sol.genotype]
        flat = [(i, j) for i, t in enumerate(trips) for j in range(len(t))]
        if len(flat) < 2:
            return sol

        # sample a few random swap candidates, pick best by heuristic
        best_swap, best_saving = None, 0

        for _ in range(min(20, len(flat))):
            (i1, j1), (i2, j2) = random.sample(flat, 2)
            if i1 == i2:
                continue

            c1, g1 = trips[i1][j1]
            c2, g2 = trips[i2][j2]

            # current detour cost for c1 in trip i1
            prev1 = trips[i1][j1-1][0] if j1 > 0 else 0
            nxt1 = trips[i1][j1+1][0] if j1 < len(trips[i1])-1 else 0
            # current detour cost for c2 in trip i2
            prev2 = trips[i2][j2-1][0] if j2 > 0 else 0
            nxt2 = trips[i2][j2+1][0] if j2 < len(trips[i2])-1 else 0

            # cost before: prev1→c1→nxt1 + prev2→c2→nxt2
            before = (rep.sp_dist(prev1, c1) + rep.sp_dist(c1, nxt1) +
                      rep.sp_dist(prev2, c2) + rep.sp_dist(c2, nxt2))
            # cost after: prev1→c2→nxt1 + prev2→c1→nxt2
            after = (rep.sp_dist(prev1, c2) + rep.sp_dist(c2, nxt1) +
                     rep.sp_dist(prev2, c1) + rep.sp_dist(c1, nxt2))

            saving = before - after
            if saving > best_saving:
                best_saving = saving
                best_swap = (i1, j1, i2, j2)

        if best_swap:
            i1, j1, i2, j2 = best_swap
            trips[i1][j1], trips[i2][j2] = trips[i2][j2], trips[i1][j1]

        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def or_opt_guided(sol: Individual, rep: PathRepresentation) -> Individual:
        """Relocate a segment of 1-3 consecutive cities to best position
        in another trip, scored by distance heuristic."""
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 2:
            return sol

        src_idx = random.randint(0, len(trips) - 1)
        if len(trips[src_idx]) < 1:
            return sol

        seg_len = random.randint(1, min(3, len(trips[src_idx])))
        start = random.randint(0, len(trips[src_idx]) - seg_len)
        segment = trips[src_idx][start:start + seg_len]
        remaining = trips[src_idx][:start] + trips[src_idx][start + seg_len:]
        trips[src_idx] = remaining

        seg_first = segment[0][0]
        seg_last = segment[-1][0]

        # find cheapest insertion across other trips
        best_trip, best_pos, best_score = None, None, float('inf')

        for t_idx, trip in enumerate(trips):
            for ins_pos in range(len(trip) + 1):
                prev = trip[ins_pos - 1][0] if ins_pos > 0 else 0
                nxt = trip[ins_pos][0] if ins_pos < len(trip) else 0

                score = (rep.sp_dist(prev, seg_first) +
                         rep.sp_dist(seg_last, nxt) -
                         rep.sp_dist(prev, nxt))

                if score < best_score:
                    best_trip, best_pos, best_score = t_idx, ins_pos, score

        trips[best_trip] = (trips[best_trip][:best_pos] +
                            segment +
                            trips[best_trip][best_pos:])
        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

class CrossoverOperators:

    @staticmethod
    def trip_crossover(p1: list, p2: list, rep: PathRepresentation) -> Individual:
        """Take random subset of trips from p1, fill remaining gold from p2."""
        # take some trips from p1
        k = random.randint(1, max(1, len(p1) // 2))
        child_trips = [list(t) for t in random.sample(p1, k)]

        # track what gold is already covered
        covered = defaultdict(float)
        for t in child_trips:
            for c, g in t:
                covered[c] += g

        # fill remaining from p2's structure
        for t in p2:
            new_t = []
            for c, g in t:
                need = rep.golds[c] - covered[c]
                if need > 0.01:
                    take = min(g, need)
                    new_t.append((c, take))
                    covered[c] += take
            if new_t:
                child_trips.append(new_t)

        # any city still missing
        for c in rep.cities:
            need = rep.golds[c] - covered.get(c, 0)
            if need > 0.01:
                child_trips.append([(c, need)])

        return Individual(child_trips, rep.eval_solution(child_trips))

    @staticmethod
    def targeted_crossover(p1: Individual, p2: Individual, rep: PathRepresentation) -> Individual:
        """Find shared city groupings between parents, disrupt them."""
        trips1 = p1.genotype
        trips2 = p2.genotype

        # build city→trip_index mapping for each parent
        group1 = {}  # city -> set of cities in same trip
        for t in trips1:
            cities_in_trip = frozenset(c for c, g in t)
            for c, _ in t:
                group1[c] = cities_in_trip

        group2 = {}
        for t in trips2:
            cities_in_trip = frozenset(c for c, g in t)
            for c, _ in t:
                group2[c] = cities_in_trip

        # find cities that are grouped together in BOTH parents
        shared_pairs = set()
        for c in rep.cities:
            if c in group1 and c in group2:
                common = group1[c] & group2[c]
                if len(common) > 1:
                    shared_pairs.add(frozenset(common))

        # start from parent1, then disrupt the shared groupings
        child_trips = [list(t) for t in trips1]

        for shared_group in shared_pairs:
            if len(shared_group) < 2 or random.random() > 0.1:
                continue

            # find which trip contains this group
            target_idx = None
            for i, t in enumerate(child_trips):
                trip_cities = {c for c, g in t}
                if shared_group.issubset(trip_cities):
                    target_idx = i
                    break

            if target_idx is None:
                continue

            # DISRUPT: pull random subset out into a separate trip
            trip = child_trips[target_idx]
            shared_list = [item for item in trip if item[0] in shared_group]
            n_pull = random.randint(1, max(1, len(shared_list) // 2))
            to_pull = set(c for c, g in random.sample(shared_list, n_pull))

            stay = [(c, g) for c, g in trip if c not in to_pull]
            pulled = [(c, g) for c, g in trip if c in to_pull]

            child_trips[target_idx] = stay

            # either make new trip or merge into a random existing trip
            if random.random() < 0.5 or len(child_trips) < 2:
                child_trips.append(pulled)
            else:
                merge_idx = random.randint(0, len(child_trips) - 1)
                child_trips[merge_idx].extend(pulled)
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


    def init_high_beta(self):
        """beta>1: individual round-trips, find best k per city."""
        sol = []
        for c in self.representation.cities:
            g, d = self.representation.golds[c], self.representation.sp_dist(0, c)
            best_k, best_cost = 1, float('inf')
            for k in range(1, 50):
                gk = g / k
                per = d + d + (self.representation.alpha * d * gk) ** self.representation.beta
                total = k * per
                if total < best_cost:
                    best_k, best_cost = k, total
                elif total > best_cost * 1.05:
                    break
            gk = g / best_k
            for _ in range(best_k):
                sol.append([(c, gk)])
        return sol

    def init_low_beta(self):
        """beta=1: nearest-neighbor multi-city tours."""
        sol, unvisited = [], set(self.representation.cities)
        while unvisited:
            trip, cur, w = [], 0, 0.0
            cap = random.uniform(0.3, 1.0) * sum(self.representation.golds[c] for c in unvisited) # random capacity (between 30%-100%) * golds of remaining cities
            while unvisited and w < cap:
                nxt = min(unvisited, key=lambda c: self.representation.sp_dist(cur, c))
                trip.append((nxt, self.representation.golds[nxt]))
                w += self.representation.golds[nxt]
                unvisited.discard(nxt)
                cur = nxt
            if trip:
                sol.append(self.representation.optimize_trip_order(trip))
        return sol

    def init_random(self):
        if self.representation.beta > 1:
            return self.init_high_beta()
        return self.init_low_beta()

    def initialize_population(self) -> list[Individual]:
        population = []
        for i in tqdm(range(self.population_size), desc="Initializing population"):
            genotype = self.init_random()
            fitness = self.representation.eval_solution(genotype)
            population.append(Individual(genotype, fitness))
        population.sort(key=lambda ind: ind.fitness)
        return population

                
    def evaluate_population(self, offspring: list[Individual]) -> list[Individual]:
        # (mu + lambda) selection: merge parents + offspring, keep best
        combined = self.current_population + offspring
        combined.sort(key=lambda ind: ind.fitness)
        return combined[:self.population_size]

    def solve(self, num_generations=100, mutation_rate=0.2):
        history = []
        # pick operators based on beta
        if self.representation.beta > 1:
            mut_ops = [MutationOperators.change_k, MutationOperators.swap,
                       MutationOperators.split_trip]
        else:
            # mut_ops = [MutationOperators.merge_trips, MutationOperators.move_city,
            #            MutationOperators.swap, MutationOperators.split_trip]

            mut_ops = [AStarGuidedOperators.insertion_mutate, AStarGuidedOperators.swap_guided,
                       AStarGuidedOperators.or_opt_guided]
            
        for generation in tqdm(range(num_generations), desc="Evolving"):
            offspring = []
            for _ in range(self.offspring_size):
                parent1 = self.selection_method(self.current_population)

                if random.random() > mutation_rate:
                    parent2 = self.selection_method(self.current_population)
                    # in solve(), track stagnation
                    # if self.representation.beta == 1 and len(history) > 20 and (abs(history[-1] - history[-20]) < 0.01 * history[-1]):
                    #     # stagnating → use targeted crossover more
                    #     child = CrossoverOperators.targeted_crossover(parent1, parent2, self.representation)
                    # else:
                    #     # normal crossover
                    child = CrossoverOperators.trip_crossover(parent1.genotype, parent2.genotype, self.representation)
                else:
                    mut_fn = random.choice(mut_ops)
                    child = mut_fn(parent1, self.representation)

                offspring.append(child)

            self.current_population = self.evaluate_population(offspring)
            history.append(self.current_population[0].fitness)

        return self.current_population, history

    def plot(self, history):
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Evolution of Best Fitness Over Generations')
        plt.grid()
        plt.show()
