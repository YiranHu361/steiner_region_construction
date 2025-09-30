import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.spatial import distance_matrix, cKDTree
from sklearn.cluster import KMeans
from scipy.optimize import differential_evolution, minimize
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
warnings.filterwarnings("ignore", message="findfont: Font family.*not found.")
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

class StrictDegreeSteinerTreeSolver:
    def __init__(self, vertices, max_steiner_points, max_degree_per_steiner, allow_direct_connection=True):
        self.vertices = vertices
        self.vertex_names = list(vertices.keys())
        self.vertex_coords = np.array(list(vertices.values()))
        self.n_vertices = len(vertices)
        self.max_steiner_points = max_steiner_points
        self.max_degree_per_steiner = max_degree_per_steiner
        self.allow_direct_connection = allow_direct_connection

        self.vertex_dist_matrix = distance_matrix(self.vertex_coords, self.vertex_coords)
        np.fill_diagonal(self.vertex_dist_matrix, np.inf)
        self.vertex_kdtree = cKDTree(self.vertex_coords)
        self.steiner_cache = {}

    def create_graph_with_cache(self, steiner_points):
        cache_key = tuple(steiner_points.flatten().tolist())
        if cache_key in self.steiner_cache:
            return self.steiner_cache[cache_key].copy()

        G = nx.Graph()
        for i, (name, coord) in enumerate(self.vertices.items()):
            G.add_node(f"v_{i}", pos=coord, type="vertex")
        steiner_coords = steiner_points.reshape(-1, 2)
        for i, coord in enumerate(steiner_coords):
            G.add_node(f"s_{i}", pos=coord, type="steiner")

        vertex_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'vertex']
        steiner_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'steiner']

        for v_node in vertex_nodes:
            v_idx = int(v_node.split('_')[1])
            v_pos = self.vertex_coords[v_idx]
            for s_node in steiner_nodes:
                s_idx = int(s_node.split('_')[1])
                s_pos = steiner_coords[s_idx]
                dist = np.linalg.norm(v_pos - s_pos)
                G.add_edge(v_node, s_node, weight=dist)

        if len(steiner_nodes) > 1:
            for i, s1 in enumerate(steiner_nodes):
                for j, s2 in enumerate(steiner_nodes[i + 1:], i + 1):
                    pos_s1 = G.nodes[s1]['pos']
                    pos_s2 = G.nodes[s2]['pos']
                    dist = np.linalg.norm(np.array(pos_s1) - np.array(pos_s2))
                    G.add_edge(s1, s2, weight=dist)

        if self.allow_direct_connection:
            for i, v1 in enumerate(vertex_nodes):
                for j, v2 in enumerate(vertex_nodes[i + 1:], i + 1):
                    G.add_edge(v1, v2, weight=self.vertex_dist_matrix[i, j])

        self.steiner_cache[cache_key] = G.copy()
        return G

    def build_degree_constrained_mst(self, G):
        mst = nx.Graph()
        visited = set()
        steiner_degrees = defaultdict(int)

        for node, data in G.nodes(data=True):
            mst.add_node(node, **data)

        start_node = next((n for n in G.nodes() if n.startswith('v_')), list(G.nodes())[0])
        visited.add(start_node)

        while len(visited) < len(G.nodes()):
            min_edge = None
            min_weight = float('inf')

            for u in visited:
                for v in G.neighbors(u):
                    if v not in visited:
                        weight = G[u][v]['weight']
                        if (v.startswith('s_') and steiner_degrees[v] >= self.max_degree_per_steiner) or \
                                (u.startswith('s_') and steiner_degrees[u] >= self.max_degree_per_steiner):
                            continue
                        if weight < min_weight:
                            min_weight = weight
                            min_edge = (u, v)

            if min_edge is None:
                for u in visited:
                    for v in G.neighbors(u):
                        if v not in visited and G[u][v]['weight'] < min_weight:
                            min_weight = G[u][v]['weight']
                            min_edge = (u, v)

            if min_edge is None:
                break

            u, v = min_edge
            mst.add_edge(u, v, weight=min_weight)
            visited.add(v)
            if u.startswith('s_'):
                steiner_degrees[u] += 1
            if v.startswith('s_'):
                steiner_degrees[v] += 1

        return mst, steiner_degrees

    def objective_function(self, steiner_points_flat):
        steiner_points = steiner_points_flat.reshape(-1, 2)
        G = self.create_graph_with_cache(steiner_points)

        try:
            mst, degree_info = self.build_degree_constrained_mst(G)
            total_length = sum(edge[2]['weight'] for edge in mst.edges(data=True))

            penalty = 0
            for node, degree in degree_info.items():
                if degree > self.max_degree_per_steiner:
                    excess = degree - self.max_degree_per_steiner
                    penalty += 1000 * (2 ** excess)

            return total_length + penalty
        except nx.NetworkXError:
            return 1e10

    def solve_global(self, bounds):
        result = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            popsize=15,
            maxiter=50,
            tol=1e-4,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        return result.x

    def solve_local(self, initial_guess, bounds):
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        )
        return result.x

    def solve_single(self, initial_steiner):
        x_min, y_min = np.min(self.vertex_coords, axis=0) - 1
        x_max, y_max = np.max(self.vertex_coords, axis=0) + 1
        bounds = [(x_min, x_max), (y_min, y_max)] * self.max_steiner_points

        global_sol = self.solve_global(bounds)
        local_sol = self.solve_local(global_sol, bounds)

        optimal_steiner = local_sol.reshape(-1, 2)
        final_G = self.create_graph_with_cache(optimal_steiner)
        final_mst, degree_info = self.build_degree_constrained_mst(final_G)
        total_length = sum(edge[2]['weight'] for edge in final_mst.edges(data=True))
        violation = sum(max(0, d - self.max_degree_per_steiner) for d in degree_info.values())

        return optimal_steiner, final_mst, total_length, violation

    def solve(self, n_attempts=3):
        best_steiner = None
        best_mst = None
        best_length = float('inf')
        best_violation = float('inf')

        initial_strategies = []
        if self.max_steiner_points > 0:
            kmeans = KMeans(n_clusters=self.max_steiner_points, n_init=10, random_state=42)
            kmeans.fit(self.vertex_coords)
            initial_strategies.append(kmeans.cluster_centers_.flatten())
        if self.max_steiner_points > 0 and self.max_steiner_points <= self.n_vertices:
            centroid = np.mean(self.vertex_coords, axis=0)
            farthest_idx = np.argsort(np.linalg.norm(self.vertex_coords - centroid, axis=1))[-self.max_steiner_points:]
            initial_strategies.append(self.vertex_coords[farthest_idx].flatten())
        for _ in range(max(1, n_attempts - len(initial_strategies))):
            x_min, y_min = np.min(self.vertex_coords, axis=0)
            x_max, y_max = np.max(self.vertex_coords, axis=0)
            random_init = np.random.uniform(low=[x_min, y_min] * self.max_steiner_points,
                                            high=[x_max, y_max] * self.max_steiner_points)
            initial_strategies.append(random_init)

        results = []
        with ThreadPoolExecutor(max_workers=min(n_attempts, 4)) as executor:
            futures = [executor.submit(self.solve_single, init) for init in initial_strategies]
            for future in as_completed(futures):
                try:
                    steiner, mst, length, violation = future.result()
                    results.append((steiner, mst, length, violation))
                    if (violation < best_violation) or (violation == best_violation and length < best_length):
                        best_violation = violation
                        best_length = length
                        best_steiner = steiner
                        best_mst = mst
                except Exception:
                    continue

        if best_steiner is None and results:
            best_steiner, best_mst, best_length, _ = results[-1]

        return best_steiner, best_mst, best_length

    def visualize(self, steiner_points, mst):
        plt.figure(figsize=(12, 10))

        vertex_coords = self.vertex_coords
        plt.scatter(vertex_coords[:, 0], vertex_coords[:, 1], c='blue', s=100, label='顶点')
        if len(steiner_points) > 0:
            plt.scatter(steiner_points[:, 0], steiner_points[:, 1], c='red', s=100, label='中转点')

        for u, v in mst.edges():
            pos_u = mst.nodes[u]['pos']
            pos_v = mst.nodes[v]['pos']
            plt.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', alpha=0.7)
            mid_x, mid_y = (pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2
            weight = mst[u][v]['weight']
            plt.text(mid_x, mid_y, f'{weight:.2f}', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        for i, (name, coord) in enumerate(self.vertices.items()):
            plt.annotate(name, (coord[0], coord[1]), xytext=(5, 5), textcoords='offset points',
                         fontsize=12, fontweight='bold')
        if len(steiner_points) > 0:
            for i, coord in enumerate(steiner_points):
                plt.annotate(f'S{i}', (coord[0], coord[1]), xytext=(-10, -10),
                             textcoords='offset points', color='red', fontweight='bold')

        plt.title('最小生成树与中转点', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    vertices = {
        'A': (5, 0),
        'B': (4, 0),
        'C': (1, 3),
        'D': (6, 5),
        'E': (4, 7),
        'F': (5, 9)
    }

    max_steiner_points = 3
    max_degree_per_steiner = 3
    allow_direct_connection = False

    solver = StrictDegreeSteinerTreeSolver(vertices, max_steiner_points, max_degree_per_steiner,
                                           allow_direct_connection)
    steiner_points, mst, total_length = solver.solve(n_attempts=3)
    solver.visualize(steiner_points, mst)

    print(f"最优中转点位置:\n{steiner_points.round(4)}")
    print(f"最小生成树总长度: {total_length:.4f}")
    print("MST边集（顶点-中转点/中转点-中转点）:")
    for u, v in mst.edges():
        print(f"  {u} <-> {v} (长度: {mst[u][v]['weight']:.4f})")
        
import numpy as np

def build_steiner_tree(points_xy,
                       max_steiner_points=None,
                       max_degree_per_steiner=3,
                       allow_direct_connection=True,
                       attempts=3):
    points_xy = np.asarray(points_xy, dtype=float)
    m = points_xy.shape[0]
    if m <= 1:
        return points_xy.copy(), []

    if max_steiner_points is None:
        max_steiner_points = max(1, min(5, int(np.ceil(m/8))))

    vertices = {f"v_{i}": (float(points_xy[i,0]), float(points_xy[i,1])) for i in range(m)}
    solver = StrictDegreeSteinerTreeSolver(
        vertices=vertices,
        max_steiner_points=max_steiner_points,
        max_degree_per_steiner=max_degree_per_steiner,
        allow_direct_connection=allow_direct_connection
    )
    steiner_pts, mst, _ = solver.solve(n_attempts=attempts)
    if steiner_pts is None:
        steiner_pts = np.empty((0,2), dtype=float)

    all_points = points_xy.copy()
    if len(steiner_pts) > 0:
        all_points = np.vstack([all_points, steiner_pts])

    edges_idx = []
    for u, v in mst.edges():
        def to_idx(lbl):
            if lbl.startswith("v_"): return int(lbl.split("_")[1])
            if lbl.startswith("s_"): return m + int(lbl.split("_")[1])
            raise ValueError(lbl)
        edges_idx.append((to_idx(u), to_idx(v)))
    return all_points, edges_idx