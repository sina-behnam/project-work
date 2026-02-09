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
        self._pos = nx.get_node_attributes(G, 'pos')
        self._n = len(G.nodes)
        self._density = kwargs.get('density', 0.5)

        # Euclidean distance matrix — always cheap, used for decisions
        coords = np.array([self._pos[i] for i in range(self._n)])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self._euc = np.sqrt(np.sum(diff**2, axis=-1))

        if self._density == 1.0:
            # direct edge = shortest path, no graph search needed ever
            self._exact_dist = self._euc
            self._use_direct = True
        else:
            # sparse: precompute exact graph distances
            self._sp = dict(nx.all_pairs_dijkstra(G, weight='dist'))
            self._use_direct = False

    def euc_dist(self, a, b):
        return self._euc[a, b]

    def sp_dist(self, a, b):
        if self._use_direct:
            return self._euc[a, b]
        return self._sp[a][0][b]

    def sp_path(self, a, b):
        if self._use_direct:
            return [a, b]
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

class AStarGuidedOperators:

    @staticmethod
    def insertion_mutate(sol: Individual, rep: PathRepresentation) -> Individual:
        trips = [list(t) for t in sol.genotype]
        if len(trips) < 1:
            return sol

        src_idx = random.randint(0, len(trips) - 1)
        if not trips[src_idx]:
            return sol
        pos = random.randint(0, len(trips[src_idx]) - 1)
        item = trips[src_idx].pop(pos)
        city, gold = item

        best_trip, best_pos, best_score = None, None, float('inf')

        for t_idx, trip in enumerate(trips):
            trip_cities = {c for c, g in trip}

            if city in trip_cities:
                # would merge — score as zero detour
                score = 0
                if score < best_score:
                    best_trip, best_pos, best_score = t_idx, -1, score
                continue

            for ins_pos in range(len(trip) + 1):
                prev = trip[ins_pos - 1][0] if ins_pos > 0 else 0
                nxt = trip[ins_pos][0] if ins_pos < len(trip) else 0
                d_detour = rep.euc_dist(prev, city) + rep.euc_dist(city, nxt)
                d_direct = rep.euc_dist(prev, nxt)
                score = d_detour - d_direct
                if score < best_score:
                    best_trip, best_pos, best_score = t_idx, ins_pos, score

        if best_pos == -1:
            # merge into existing entry
            trips[best_trip] = [(c, g + gold) if c == city else (c, g)
                                for c, g in trips[best_trip]]
        else:
            trips[best_trip].insert(best_pos, item)

        trips = [t for t in trips if t]
        return Individual(trips, rep.eval_solution(trips))

    @staticmethod
    def swap_guided(sol: Individual, rep: PathRepresentation) -> Individual:
        trips = [list(t) for t in sol.genotype]
        flat = [(i, j) for i, t in enumerate(trips) for j in range(len(t))]
        if len(flat) < 2:
            return sol
    
        best_swap, best_saving = None, 0
    
        for _ in range(min(20, len(flat))):
            (i1, j1), (i2, j2) = random.sample(flat, 2)
            if i1 == i2:
                continue
            c1, g1 = trips[i1][j1]
            c2, g2 = trips[i2][j2]
    
            prev1 = trips[i1][j1-1][0] if j1 > 0 else 0
            nxt1 = trips[i1][j1+1][0] if j1 < len(trips[i1])-1 else 0
            prev2 = trips[i2][j2-1][0] if j2 > 0 else 0
            nxt2 = trips[i2][j2+1][0] if j2 < len(trips[i2])-1 else 0
    
            before = (rep.euc_dist(prev1, c1) + rep.euc_dist(c1, nxt1) +
                      rep.euc_dist(prev2, c2) + rep.euc_dist(c2, nxt2))
            after = (rep.euc_dist(prev1, c2) + rep.euc_dist(c2, nxt1) +
                     rep.euc_dist(prev2, c1) + rep.euc_dist(c1, nxt2))
    
            saving = before - after
            if saving > best_saving:
                best_saving = saving
                best_swap = (i1, j1, i2, j2)
    
        if best_swap:
            i1, j1, i2, j2 = best_swap
            c1, g1 = trips[i1][j1]
            c2, g2 = trips[i2][j2]
            trip1_cities = {c for c, g in trips[i1]} - {c1}
            trip2_cities = {c for c, g in trips[i2]} - {c2}
    
            if c2 in trip1_cities:
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
    def or_opt_guided(sol: Individual, rep: PathRepresentation) -> Individual:
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

        best_trip, best_pos, best_score = None, None, float('inf')

        for t_idx, trip in enumerate(trips):
            for ins_pos in range(len(trip) + 1):
                prev = trip[ins_pos - 1][0] if ins_pos > 0 else 0
                nxt = trip[ins_pos][0] if ins_pos < len(trip) else 0
                score = (rep.euc_dist(prev, seg_first) +
                         rep.euc_dist(seg_last, nxt) -
                         rep.euc_dist(prev, nxt))
                if score < best_score:
                    best_trip, best_pos, best_score = t_idx, ins_pos, score

        # merge segment into target trip with duplicate check
        target = trips[best_trip]
        target_cities = {c for c, g in target}

        insert_items = []
        for city, gold in segment:
            if city in target_cities:
                target = [(c, g + gold) if c == city else (c, g)
                          for c, g in target]
            else:
                insert_items.append((city, gold))
                target_cities.add(city)

        # insert non-duplicate items at best position
        trips[best_trip] = target[:best_pos] + insert_items + target[best_pos:]
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
                   selection_method : callable = None,
                   with_init_optmization : bool = False
                   ):

        self.representation = problem_rep
        
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.selection_method = selection_method
        self.with_init_optmization = with_init_optmization

        # Initialize the current population
        self.current_population = self.initialize_population()


    def init_high_beta(self):
        """beta>1: individual round-trips, find best k per city."""
        sol = []
        for c in self.representation.cities:
            g, d = self.representation.golds[c], self.representation.euc_dist(0, c)
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
                nxt = min(unvisited, key=lambda c: self.representation.euc_dist(cur, c))
                trip.append((nxt, self.representation.golds[nxt]))
                w += self.representation.golds[nxt]
                unvisited.discard(nxt)
                cur = nxt
            if trip:
                if self.with_init_optmization:
                    sol.append(self.representation.optimize_trip_order(trip))
                else:
                    sol.append(trip) # initialize faster without optimizing trip order leaving the optimization to the evolutionary process ! 
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

    def solve(self, num_generations=100, mutation_rate=0.3):
        history = []
        stagnation_counter = 0
    
        if self.representation.beta > 1:
            # β>1: trip splitting dominates
            mut_ops = [
                (MutationOperators.change_k, 0.4),      # primary: adjust splits
                (MutationOperators.split_trip, 0.2),     # break trips apart
                (MutationOperators.swap, 0.2),           # shuffle stops
                (AStarGuidedOperators.insertion_mutate, 0.2),  # smart relocation
            ]
        else:
            # β=1: route structure dominates
            mut_ops = [
                (AStarGuidedOperators.insertion_mutate, 0.25),  # smart city relocation
                (AStarGuidedOperators.swap_guided, 0.25),       # smart inter-trip swap
                (AStarGuidedOperators.or_opt_guided, 0.2),      # smart segment move
                (MutationOperators.merge_trips, 0.15),     # combine trips
                (MutationOperators.move_city, 0.15),       # random relocation
            ]
    
        ops, weights = zip(*mut_ops)
    
        for generation in tqdm(range(num_generations), desc="Evolving"):
            offspring = []
    
            for _ in range(self.offspring_size):
                parent1 = self.selection_method(self.current_population)
    
                # phase 1: normal evolution
                if stagnation_counter < 20:
                    if random.random() > mutation_rate:
                        parent2 = self.selection_method(self.current_population)
                        child = CrossoverOperators.trip_crossover(
                            parent1.genotype, parent2.genotype, self.representation)
                    else:
                        mut_fn = random.choices(ops, weights=weights, k=1)[0]
                        child = mut_fn(parent1, self.representation)
    
                # phase 2: stagnating → targeted disruption
                else:
                    r = random.random()
                    if r < 0.4:
                        # targeted crossover to break converged structure
                        parent2 = self.selection_method(self.current_population)
                        child = CrossoverOperators.targeted_crossover(
                            parent1, parent2, self.representation)
                    elif r < 0.7:
                        # double mutation for bigger jumps
                        mut_fn = random.choices(ops, weights=weights, k=1)[0]
                        child = mut_fn(parent1, self.representation)
                        mut_fn = random.choices(ops, weights=weights, k=1)[0]
                        child = mut_fn(child, self.representation)
                    else:
                        # inject fresh random individual
                        genotype = self.init_random()
                        child = Individual(genotype, self.representation.eval_solution(genotype))
    
                offspring.append(child)
    
            self.current_population = self.evaluate_population(offspring)
            history.append(self.current_population[0].fitness)
    
            # track stagnation
            if len(history) > 10 and abs(history[-1] - history[-10]) < 0.001 * abs(history[-10]):
                stagnation_counter += 1
            else:
                stagnation_counter = 0
    
        return self.current_population, history

    def plot(self, history):
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Evolution of Best Fitness Over Generations')
        plt.grid()
        plt.show()
