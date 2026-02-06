from problem import Problem
import networkx as nx
import logging
from itertools import combinations

class MySolution(Problem):

    def simple_tsp(self):
        G = self.graph.copy()

        tour = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='dist')
        # logging.info(f"TSP Tour: {tour}")

        total_cost = 0
        current_gold = 0
        for i in range(len(tour)-1):
            if tour[i] != 0:
                gold = G.nodes[tour[i]]['gold']
                current_gold += gold

            segment = nx.shortest_path(G, source=tour[i], target=tour[i+1], weight='dist')
                
            total_cost += self.cost(segment, current_gold)

        logging.info(f"Total Cost of TSP-based solution: {total_cost:.2f}")
        return total_cost
    
    def sort_cities_by_gold(self):
        G = self.graph.copy()
        cities = [(node, G.nodes[node]['gold']) for node in G.nodes if node != 0]
        cities.sort(key=lambda x: x[1], reverse=True)
        
        total_cost = 0
        for city, gold in cities:
            path_to_city = nx.shortest_path(G, source=0, target=city, weight='dist')
            path_back = list(reversed(path_to_city))
            
            total_cost += self.cost(path_to_city, 0)
            total_cost += self.cost(path_back, gold)

        logging.info(f"Total Cost of Gold-based solution: {total_cost:.2f}")
        return total_cost
    
import heapq
import math
from dataclasses import dataclass
import numpy as np
import networkx as nx
import logging

class BucketedPickupSolver(Problem):
    @dataclass(frozen=True)
    class State:
        pos: int                  # current city
        remaining: tuple          # bucket counts per city 1..n-1
        carried: int              # carried buckets

    bucket_labels = ["very_light", "light", "heavy", "very_heavy"]

    def _bucketize_gold(self, golds):
        q1, q2, q3 = np.quantile(golds, [0.25, 0.5, 0.75])
        def b(v):
            if v <= q1: return 0
            if v <= q2: return 1
            if v <= q3: return 2
            return 3
        return {i + 1: b(g) for i, g in enumerate(golds[1:])}

    def _all_pairs_paths(self, G):
        # Precompute shortest paths by distance for every pair
        return {
            (src, dst): path
            for src, paths in nx.all_pairs_dijkstra_path(G, weight="dist")
            for dst, path in paths.items()
        }

    def solve_bucketed(
        self,
        bucket_size=None,
        pickup_mode="all_or_one",  # choices per city: ["all"] or ["one","all"]
    ):
        G = self.graph
        n = len(G)
        gold = [G.nodes[i]["gold"] if i != 0 else 0.0 for i in range(n)]
        if bucket_size is None:
            bucket_size = max(1.0, np.quantile(gold[1:], 0.25))  # default granularity
        buckets = self._bucketize_gold(gold)
        sp = self._all_pairs_paths(G)

        remaining_counts = tuple(math.ceil(g / bucket_size) for g in gold[1:])
        start = self.State(0, remaining_counts, 0)

        pq = [(0.0, 0, start)]
        counter = 0  # tie-breaker to avoid comparing State in heap
        best = {start: 0.0}
        parent = {}
        move_leg = {}  # state -> (prev_state, path_used)

        def is_goal(s: "BucketedPickupSolver.State"):
            return s.pos == 0 and s.carried == 0 and all(r == 0 for r in s.remaining)

        while pq:
            acc, _, state = heapq.heappop(pq)
            if acc != best[state]:
                continue
            if is_goal(state):
                # reconstruct walk
                walk = [0]
                trace = state
                while trace in parent:
                    prev = parent[trace]
                    leg = move_leg[trace]
                    walk = leg[:-1] + walk
                    trace = prev
                logging.info(f"Bucketed state-search cost: {acc:.2f}")
                return acc, walk, bucket_size

            # Drop-off at depot (free, only changes carried)
            if state.pos == 0 and state.carried > 0:
                dropped = self.State(0, state.remaining, 0)
                if acc < best.get(dropped, float("inf")):
                    best[dropped] = acc
                    parent[dropped] = state
                    move_leg[dropped] = [0]
                    counter += 1
                    heapq.heappush(pq, (acc, counter, dropped))
                # continue exploring moves from current state as well

            # Order remaining cities by category (light first) to bias search
            remaining_cities = [i + 1 for i, r in enumerate(state.remaining) if r > 0]
            remaining_cities.sort(key=lambda c: buckets[c])

            for city in remaining_cities:
                idx = city - 1
                rem_here = state.remaining[idx]
                take_choices = [rem_here] if pickup_mode == "all" else [1, rem_here]
                for take in take_choices:
                    new_remaining = list(state.remaining)
                    new_remaining[idx] -= take
                    leg = sp[(state.pos, city)]
                    weight_now = state.carried * bucket_size
                    step_cost = self.cost(leg, weight_now)
                    new_state = self.State(
                        city,
                        tuple(new_remaining),
                        state.carried + take,
                    )
                    new_cost = acc + step_cost
                    if new_cost < best.get(new_state, float("inf")):
                        best[new_state] = new_cost
                        parent[new_state] = state
                        move_leg[new_state] = leg
                        counter += 1
                        heapq.heappush(pq, (new_cost, counter, new_state))

            # Option: move to depot even without finishing pickups, to drop later
            if state.pos != 0:
                leg = sp[(state.pos, 0)]
                step_cost = self.cost(leg, state.carried * bucket_size)
                new_state = self.State(0, state.remaining, state.carried)
                new_cost = acc + step_cost
                if new_cost < best.get(new_state, float("inf")):
                    best[new_state] = new_cost
                    parent[new_state] = state
                    move_leg[new_state] = leg
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, new_state))

        raise RuntimeError("No solution found (graph should be connected).")
