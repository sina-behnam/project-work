# beam_search_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import networkx as nx

from problem import Problem


@dataclass(frozen=True)
class State:
    pos: int
    remaining_mask: int  # bit i = 1 means city i still has gold to pick (i>=1)
    load: float          # current carried gold


@dataclass(frozen=True)
class Action:
    kind: str            # "pickup" or "return"
    target: int
    path: Tuple[int, ...]  # shortest path nodes from old pos -> target


@dataclass
class Node:
    f: float
    g: float
    state: State
    parent: Optional["Node"]
    action: Optional[Action]


class BeamSearchCollector(Problem):
    """
    Beam search collector that *extends* Problem but clones an existing Problem instance
    (so we solve on the same graph).
    """

    def __init__(
        self,
        problem: Problem,
        *,
        beam_width: int = 200,
        candidate_k: Optional[int] = 10,
        heuristic_weight: float = 0.25,
    ):
        # Do NOT call Problem.__init__ (it would generate a new random instance).
        # Instead copy the underlying data from the provided problem.
        self._graph = problem.graph          # copy
        self._alpha = problem.alpha
        self._beta = problem.beta

        self.beam_width = beam_width
        self.candidate_k = candidate_k
        self.heuristic_weight = heuristic_weight

        self.n = self._graph.number_of_nodes()
        self.gold = [float(self._graph.nodes[i]["gold"]) for i in range(self.n)]

        # remaining gold exists for cities 1..n-1 (city 0 is depot)
        self.all_mask = 0
        for i in range(1, self.n):
            if self.gold[i] > 0:
                self.all_mask |= (1 << i)

        # Precompute all-pairs shortest paths and distances (by 'dist')
        self._paths: List[Dict[int, List[int]]] = [dict() for _ in range(self.n)]
        self._dists: List[Dict[int, float]] = [dict() for _ in range(self.n)]
        for s in range(self.n):
            dist_map, path_map = nx.single_source_dijkstra(self._graph, source=s, weight="dist")
            self._paths[s] = path_map
            self._dists[s] = dist_map

    def _initial_state(self) -> State:
        return State(pos=0, remaining_mask=self.all_mask, load=0.0)

    def _is_goal(self, st: State) -> bool:
        return st.pos == 0 and st.load == 0.0 and st.remaining_mask == 0

    def _remaining_cities(self, mask: int) -> List[int]:
        # cities are 1..n-1
        out = []
        for i in range(1, self.n):
            if (mask >> i) & 1:
                out.append(i)
        return out

    def _heuristic(self, st: State) -> float:
        """
        Simple guiding heuristic (not necessarily admissible):
        - If carrying load away from depot, add estimated cost to return to depot now.
        - For each remaining city, add the baseline-like "go empty from 0 and return carrying gold_i".
        """
        h = 0.0

        if st.pos != 0 and st.load > 0:
            h += self.cost(self._paths[st.pos][0], st.load)

        for i in self._remaining_cities(st.remaining_mask):
            # baseline-style estimate from depot
            h += self.cost(self._paths[0][i], 0.0)
            h += self.cost(self._paths[i][0], self.gold[i])

        return h

    def _successors(self, st: State) -> List[Tuple[float, State, Action]]:
        """
        Returns list of (step_cost, new_state, action).
        step_cost is computed using cost(shortest_path, current_load).
        """
        succ: List[Tuple[float, State, Action]] = []

        remaining = self._remaining_cities(st.remaining_mask)

        # Optional branching reduction: consider only k nearest remaining cities
        if self.candidate_k is not None and len(remaining) > self.candidate_k:
            remaining.sort(key=lambda c: self._dists[st.pos][c])
            remaining = remaining[: self.candidate_k]

        # 1) Go pick all gold from a remaining city
        for city in remaining:
            path = tuple(self._paths[st.pos][city])
            step_cost = self.cost(path, st.load)

            new_mask = st.remaining_mask & ~(1 << city)
            new_load = st.load + self.gold[city]
            new_state = State(pos=city, remaining_mask=new_mask, load=new_load)

            succ.append((step_cost, new_state, Action("pickup", city, path)))

        # 2) Return to depot (only if carrying something and not already at depot)
        if st.pos != 0 and st.load > 0:
            path = tuple(self._paths[st.pos][0])
            step_cost = self.cost(path, st.load)

            new_state = State(pos=0, remaining_mask=st.remaining_mask, load=0.0)
            succ.append((step_cost, new_state, Action("return", 0, path)))

        return succ

    @staticmethod
    def _reconstruct_actions(goal_node: Node) -> List[Action]:
        actions: List[Action] = []
        cur = goal_node
        while cur.parent is not None and cur.action is not None:
            actions.append(cur.action)
            cur = cur.parent
        actions.reverse()
        return actions

    @staticmethod
    def actions_to_route(actions: List[Action]) -> List[int]:
        """
        Concatenate the node paths in actions into one full route.
        Ensures we don't duplicate the joining node.
        """
        route: List[int] = []
        for a in actions:
            if not route:
                route.extend(list(a.path))
            else:
                # avoid duplicating the first node (same as last of current route)
                route.extend(list(a.path)[1:])
        return route

    def solve(self, *, max_steps: Optional[int] = None) -> Tuple[float, List[Action], List[int]]:
        """
        Beam search:
        - keeps top `beam_width` nodes by f = g + heuristic_weight * h
        Returns: (best_cost, actions, full_route_nodes)
        """
        if max_steps is None:
            # Safe upper bound: pick every city then return each time
            max_steps = 2 * (self.n - 1) + 2

        start = self._initial_state()
        h0 = self._heuristic(start)
        beam: List[Node] = [Node(f=0.0 + self.heuristic_weight * h0, g=0.0, state=start, parent=None, action=None)]

        best_goal: Optional[Node] = None

        for _depth in range(max_steps):
            candidates: List[Node] = []

            for node in beam:
                if self._is_goal(node.state):
                    best_goal = node
                    break

                for step_cost, new_state, action in self._successors(node.state):
                    new_g = node.g + step_cost
                    new_h = self._heuristic(new_state)
                    new_f = new_g + self.heuristic_weight * new_h
                    candidates.append(Node(f=new_f, g=new_g, state=new_state, parent=node, action=action))

            if best_goal is not None:
                break

            if not candidates:
                break

            # keep only best beam_width by f
            beam = heapq.nsmallest(self.beam_width, candidates, key=lambda nd: nd.f)

        # pick best goal if found, otherwise best partial
        if best_goal is None:
            best_goal = min(beam, key=lambda nd: nd.f)

        actions = self._reconstruct_actions(best_goal)
        route = self.actions_to_route(actions)
        return best_goal.g, actions, route


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    p = Problem(num_cities=25, alpha=1.0, beta=1.2, density=0.4, seed=42)

    solver = BeamSearchCollector(
        p,
        beam_width=300,
        candidate_k=10,
        heuristic_weight=0.25,
    )

    best_cost, actions, route = solver.solve()

    print("Baseline cost:", p.baseline())
    print("Beam best cost:", best_cost)
    print("Number of actions:", len(actions))
    print("Route (nodes):", route)
