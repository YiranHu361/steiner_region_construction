import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import importlib, importlib.util, sys, os

diff = 1

def fun(SEED) :
    # We will try to import a module named MST_NEW or load it from
    # a relative path “experiment/MST_NEW.py”. A helper function
    MST_NEW_RELATIVE_PATH = os.path.join("experiment", "MST_NEW.py")
    MST_NEW_MODULE_NAME   = "MST_NEW" 

    # These values control the random instance, domain, and ranges
    # for costs. Change SEED so we can see different graphs!!
    N = 125
    DOMAIN = (0.0, 1.0, 0.0, 1.0)

    DSC_RANGE = (8000, 12000)
    IGC_RANGE = (3000, 6000)
    UNIT_COST_RANGE = (250.0, 600.0)

    NUM_FLAT_REGIONS = 2
    REGION_R_RANGE = (0.18, 0.25)
    REGION_PAD = 0.01
    REGION_GAP = 0.002

    ST_COST_RANGE = (150.0, 500.0)
    SUBSTATION_COST_RANGE = (6000.0, 12000.0)

    # We sample N points in the unit square and assign costs:
    # DSC (decentralized cost), IGC (internal centralized cost), and a
    # global UNIT_COST for per-unit edge length. MVmax is the feasibility
    np.random.seed(SEED)
    xmin, xmax, ymin, ymax = DOMAIN

    xs = np.random.rand(N) * (xmax - xmin) + xmin
    ys = np.random.rand(N) * (ymax - ymin) + ymin
    points = np.stack([xs, ys], axis=1)

    DSC = np.random.uniform(*DSC_RANGE, size=N)
    IGC = np.random.uniform(*IGC_RANGE, size=N)
    rng_sub = np.random.default_rng(SEED + 5150)
    SUBSTATION_COST = float(rng_sub.uniform(*SUBSTATION_COST_RANGE))
    UNIT_COST = float(np.random.uniform(*UNIT_COST_RANGE))

    DV = DSC - IGC
    MVmax = np.clip(DV / UNIT_COST, 0.0, None)

    # We use a standard DSU to maintain components while scanning
    # edges in nondecreasing distance order.
    class DSU:
        def __init__(self, n):
            self.p = list(range(n)); self.r = [0]*n
        def find(self, x):
            while self.p[x] != x:
                self.p[x] = self.p[self.p[x]]
                x = self.p[x]
            return x
        def union(self, a, b):
            ra, rb = self.find(a), self.find(b)
            if ra == rb: return ra
            if self.r[ra] < self.r[rb]:
                self.p[ra] = rb; return rb
            elif self.r[rb] < self.r[ra]:
                self.p[rb] = ra; return ra
            else:
                self.p[rb] = ra; self.r[ra]+=1; return ra


    # We consider all pairs (i,j), compute distances, and sort them.
    edges_sorted = []
    for i in range(N):
        for j in range(i+1, N):
            d = float(np.hypot(points[i,0]-points[j,0], points[i,1]-points[j,1]))
            edges_sorted.append((i, j, d))
    edges_sorted.sort(key=lambda e: e[2])

    # We scan edges from short to long. An edge (i,j) is feasible if
    # both nodes can reach each other under MV constraints. If adding
    # it merges two components for the first time, we compare the gain
    # from centralizing endpoints versus staying decentralized.
    dsu = DSU(N)
    comp_nodes = {i: {i} for i in range(N)}
    comp_cent = {i: False for i in range(N)}
    kept = []   # edges that we accept: (u, v, dist)
    total_cost = float(np.sum(DSC))

    for i, j, d in edges_sorted:
        if not (MVmax[i] >= d and MVmax[j] >= d):
            continue
        ri, rj = dsu.find(i), dsu.find(j)
        if ri == rj: 
            continue

        # --- New delta: line cost + (node switching, if any) + substation effect ---
        delta = UNIT_COST * d
        if not comp_cent[ri] and not comp_cent[rj]:
            delta += (IGC[i] - DSC[i]) + (IGC[j] - DSC[j]) + SUBSTATION_COST

        elif comp_cent[ri] and comp_cent[rj]:
            delta += -SUBSTATION_COST

        else:
            if not comp_cent[ri]:
                delta += (IGC[i] - DSC[i])
            else:
                delta += (IGC[j] - DSC[j])

        if delta < 0.0:
            new_root = dsu.union(ri, rj)
            old_root = rj if new_root == ri else ri

            comp_nodes[new_root] = comp_nodes.get(new_root, set()) | comp_nodes.get(old_root, set())
            comp_nodes.pop(old_root, None)

            comp_cent[new_root] = True
            comp_cent.pop(old_root, None)

            kept.append((i, j, d))
            total_cost += delta

    # Any endpoint that appears in a kept edge is flagged as centralized.
    centralized = set()
    for u, v, _ in kept:
        centralized.add(u); centralized.add(v)

    # For each node, record the shortest feasible neighbor distance dmin,
    # and a local margin UNIT_COST*dmin - (DSC-IGC). These help sanity-check
    # the instance but do not affect the logic that follows.
    dmin = np.full(N, np.inf)
    for i in range(N):
        xi, yi = points[i]
        for j in range(N):
            if i == j: continue
            d = float(np.hypot(points[j,0]-xi, points[j,1]-yi))
            if d <= MVmax[i] and d <= MVmax[j]:
                if d < dmin[i]: dmin[i] = d
    local_margin = UNIT_COST * dmin - (DSC - IGC)

    # nodes_df collects per-node features and the centralized flag.
    # edges_df lists every kept edge with its length and line cost.
    nodes_df = pd.DataFrame({
        "id": np.arange(N, dtype=int),
        "x": xs, "y": ys,
        "DSC": DSC, "IGC": IGC,
        "DV=DSC-IGC": DSC - IGC,
        "MVmax": MVmax,
        "d_feasible_min": dmin,
        "local_margin": local_margin,
        "centralized": [int(i in centralized) for i in range(N)],
        "UNIT_COST": UNIT_COST
    })
    edges_records = [{"u": u, "v": v, "dist": d, "egc": UNIT_COST * d} for (u, v, d) in kept]
    edges_df = pd.DataFrame(edges_records, columns=["u", "v", "dist", "egc"])  # columns set even if empty

    # This plot shows kept edges and marks centralized vs. decentralized
    # nodes. It is a state before any regional Steiner rewiring.
    nodes_path = "m2_nodes.csv"
    edges_path = "m2_edges.csv"
    plot_path = "m1_plot.png"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    fig = plt.figure(figsize=(7,7))
    for _, row in edges_df.iterrows():
        u, v = int(row.u), int(row.v)
        x1, y1 = points[u]; x2, y2 = points[v]
        plt.plot([x1, x2], [y1, y2])
    cen_mask = nodes_df["centralized"].values.astype(bool)
    dec_mask = ~cen_mask
    plt.scatter(points[dec_mask,0], points[dec_mask,1], s=28, label="Decentralized")
    plt.scatter(points[cen_mask,0], points[cen_mask,1], s=40, label="Centralized")
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)

    # We now place k non-overlapping circular regions in the domain.
    # We will later consider centralized nodes inside each region as
    # “terminals” that can be re-wired by a Steiner-like topology.

    # Describes how we sample k circles inside DOMAIN with a minimum gap.
    # Ensures circles neither overlap nor touch; may raise if impossible.
    def sample_flat_regions_nonoverlap(k=NUM_FLAT_REGIONS,
                                    r_range=REGION_R_RANGE,
                                    pad=REGION_PAD,
                                    gap=REGION_GAP,
                                    seed=None,
                                    max_tries=20000):
        rng = np.random.default_rng(seed)
        xmin, xmax, ymin, ymax = DOMAIN
        placed = []  

        tries = 0
        while len(placed) < k and tries < max_tries:
            tries += 1
            r = float(rng.uniform(*r_range))
            cx = float(rng.uniform(xmin + pad + r, xmax - pad - r))
            cy = float(rng.uniform(ymin + pad + r, ymax - pad - r))

            ok = True
            for (ocx, ocy, orad) in placed:
                if (cx - ocx)**2 + (cy - ocy)**2 < (r + orad + gap)**2:
                    ok = False
                    break
            if ok:
                placed.append((cx, cy, r))

        if len(placed) < k:
            raise RuntimeError(
                f"无法放置 {k} 个不重叠圆（已放置 {len(placed)} 个）。"
                "调小 REGION_R_RANGE 或减少 NUM_FLAT_REGIONS，或降低 REGION_GAP。"
            )

        rows = [{"region_id": rid, "cx": cx, "cy": cy, "r": r} for rid, (cx, cy, r) in enumerate(placed)]
        return pd.DataFrame(rows, columns=["region_id", "cx", "cy", "r"])

    # Returns a boolean: whether nodes lying inside the circle (cx,cy,r).
    # Uses squared distances to avoid repeated square roots.
    def points_in_region(cx, cy, r):
        dx = xs - cx
        dy = ys - cy
        return (dx*dx + dy*dy) <= (r*r)

    # Generate and save the regions, then list centralized points inside.
    regions_df = sample_flat_regions_nonoverlap(seed=SEED + 777)
    regions_df.to_csv("flat_regions.csv", index=False)

    centralized_mask = nodes_df["centralized"].astype(bool).to_numpy()
    rows = []
    for row in regions_df.itertuples(index=False):
        mask_inside = points_in_region(row.cx, row.cy, row.r)
        ids = np.where(mask_inside & centralized_mask)[0]
        for nid in ids:
            rows.append({
                "region_id": int(row.region_id),
                "cx": float(row.cx),
                "cy": float(row.cy),
                "r": float(row.r),
                "node_id": int(nid),
                "x": float(xs[nid]),
                "y": float(ys[nid]),
                "DSC": float(DSC[nid]),
                "IGC": float(IGC[nid]),
                "UNIT_COST": float(UNIT_COST)
            })
    centralized_in_regions_df = pd.DataFrame(
        rows,
        columns=["region_id","cx","cy","r","node_id","x","y","DSC","IGC","UNIT_COST"]
    )
    centralized_in_regions_df.to_csv("centralized_in_regions.csv", index=False)

    # This second figure overlays the region circles on top of the
    # baseline network so you can see where terminals live.
    theta = np.linspace(0.0, 2.0*np.pi, 400)
    fig = plt.figure(figsize=(7,7))
    for _, erow in edges_df.iterrows():
        u, v = int(erow.u), int(erow.v)
        x1, y1 = points[u]; x2, y2 = points[v]
        plt.plot([x1, x2], [y1, y2])
    cen_mask = nodes_df["centralized"].values.astype(bool)
    dec_mask = ~cen_mask
    plt.scatter(points[dec_mask,0], points[dec_mask,1], s=28, label="Decentralized")
    plt.scatter(points[cen_mask,0], points[cen_mask,1], s=40, label="Centralized")
    for reg in regions_df.itertuples(index=False):
        cx, cy, r = reg.cx, reg.cy, reg.r
        plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta))
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig("m1_plot_with_regions.png", dpi=160)
    plt.close(fig)

    # Returns edges-as-coordinates from a Prim MST over coord array.
    def _prim_mst_edges(coords: np.ndarray):
        m = coords.shape[0]
        if m < 2:
            return []
        used = np.zeros(m, dtype=bool)
        used[0] = True
        dist = np.full(m, np.inf)
        parent = np.full(m, -1, dtype=int)
        for j in range(1, m):
            dx = coords[j,0]-coords[0,0]
            dy = coords[j,1]-coords[0,1]
            dist[j] = (dx*dx + dy*dy)**0.5
            parent[j] = 0
        edges = []
        for _ in range(m-1):
            j = int(np.argmin(dist + used*1e30))
            if used[j]:
                break
            used[j] = True
            i = parent[j]
            edges.append(((coords[i,0], coords[i,1]), (coords[j,0], coords[j,1])))
            for k in range(m):
                if not used[k]:
                    dx = coords[k,0]-coords[j,0]
                    dy = coords[k,1]-coords[j,1]
                    d = (dx*dx + dy*dy)**0.5
                    if d < dist[k]:
                        dist[k] = d
                        parent[k] = j
        return edges

    # Tries multiple ways to call the MST_NEW implementation.
    def _call_user_mst_new(terminals_xy: np.ndarray):
        """
        Try to call user's MST_NEW via multiple routes:
        1) wrappers in globals(),
        2) import experiment.MST_NEW (package import),
        3) load from relative file path MST_NEW_RELATIVE_PATH,
        4) import MST_NEW (plain),
        5) as a last resort, call StrictDegreeSteinerTreeSolver inside the loaded module.
        Returns (all_points, edges_idx) or a list of ((x1,y1),(x2,y2)) or None.
        """
        # 0) wrappers in globals
        for name in ("build_steiner_tree", "steiner_tree", "run", "compute"):
            fn = globals().get(name)
            if callable(fn):
                try:
                    print(f"[MST adapter] using globals.{name}")
                    return fn(terminals_xy)
                except Exception as e:
                    print(f"[MST adapter] globals.{name} failed: {e}")

        def _try_wrappers(mod):
            for name in ("build_steiner_tree", "steiner_tree", "run", "compute"):
                fn = getattr(mod, name, None)
                if callable(fn):
                    try:
                        print(f"[MST adapter] using {mod.__name__}.{name}")
                        return fn(terminals_xy)
                    except Exception as e:
                        print(f"[MST adapter] {mod.__name__}.{name} failed: {e}")
            return None

        mod = None
        try:
            mod = importlib.import_module("MST_NEW")
            sys.modules[MST_NEW_MODULE_NAME] = mod  # also alias as MST_NEW if needed
        except Exception as e:
            print(f"[MST adapter] import experiment.MST_NEW failed: {e}")

        # load from relative file path (experiment/MST_NEW.py)
        if mod is None:
            # resolve path relative to this script if possible, else CWD
            base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
            cand = os.path.join(base_dir, MST_NEW_RELATIVE_PATH)
            if os.path.exists(cand):
                try:
                    spec = importlib.util.spec_from_file_location(MST_NEW_MODULE_NAME, cand)
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)
                    sys.modules[MST_NEW_MODULE_NAME] = mod
                    print(f"[MST adapter] loaded from {cand}")
                except Exception as e:
                    print(f"[MST adapter] load from {cand} failed: {e}")

        # plain import MST_NEW (in case it’s on PYTHONPATH)
        if mod is None:
            try:
                mod = importlib.import_module(MST_NEW_MODULE_NAME)
            except Exception as e:
                print(f"[MST adapter] import {MST_NEW_MODULE_NAME} failed: {e}")

        if mod is None:
            return None

        # try wrappers inside the module
        res = _try_wrappers(mod)
        if res is not None:
            return res

        # no wrappers: use the class StrictDegreeSteinerTreeSolver if present
        try:
            cls = getattr(mod, "StrictDegreeSteinerTreeSolver")
            m = int(terminals_xy.shape[0])
            k = max(1, min(6, int(np.ceil(m/6))))  # heuristic for max_steiner_points
            solver = cls(
                vertices={f"v_{i}": (float(x), float(y)) for i, (x, y) in enumerate(terminals_xy)},
                max_steiner_points=k,
                max_degree_per_steiner=3,
                allow_direct_connection=False  # encourage Steiner junctions
            )
            steiner_pts, mst, _ = solver.solve(n_attempts=3)
            if steiner_pts is None:
                steiner_pts = np.empty((0, 2), dtype=float)

            # Convert to (all_points, edges_idx)
            all_points = terminals_xy if len(steiner_pts) == 0 else np.vstack([terminals_xy, steiner_pts])
            edges_idx = []
            for u, v in mst.edges():
                def _to_idx(lbl: str) -> int:
                    if lbl.startswith("v_"): return int(lbl.split("_")[1])
                    if lbl.startswith("s_"): return m + int(lbl.split("_")[1])
                    raise ValueError(f"Unknown node label: {lbl}")
                edges_idx.append((_to_idx(u), _to_idx(v)))

            print(f"[MST adapter] used StrictDegreeSteinerTreeSolver (k={k})")
            return all_points, edges_idx
        except Exception as e:
            print(f"[MST adapter] class path failed: {e}")
            return None

    # Wraps calling MST_NEW and normalizes its output to coordinate edges.
    # If a wrapper returns (points, edges_idx), we expand to segments.
    def _run_steiner_via_mst_new(terminals_xy: np.ndarray):
        res = _call_user_mst_new(terminals_xy)
        if res is not None:
            # 期望：result = (all_points: ndarray (M,2), edges_idx: List[(u,v)])
            if (isinstance(res, (list, tuple)) and len(res) >= 2 and
                isinstance(res[0], np.ndarray) and np.ndim(res[0]) == 2 and
                isinstance(res[1], (list, tuple))):
                all_pts = np.asarray(res[0], dtype=float)
                edges_idx = [(int(a), int(b)) for (a,b) in res[1]]
                edges_coords = [((all_pts[u,0], all_pts[u,1]), (all_pts[v,0], all_pts[v,1]))
                                for (u,v) in edges_idx]
                m = terminals_xy.shape[0]
                steiner_pts = all_pts[m:] if all_pts.shape[0] > m else np.empty((0,2))
                return edges_coords, steiner_pts
            # 另一种可能：直接返回坐标对列表
            if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple)):
                first = res[0]
                if (isinstance(first, (list, tuple)) and len(first) == 2 and
                    isinstance(first[0], (list, tuple, np.ndarray)) and
                    isinstance(first[1], (list, tuple, np.ndarray))):
                    edges_coords = [ (tuple(map(float, a)), tuple(map(float, b))) for (a,b) in res ]
                    return edges_coords, np.empty((0,2))
        # 回退
        return _prim_mst_edges(terminals_xy), np.empty((0,2))

    # We hide baseline edges fully inside eligible regions, then plot
    # only the outside edges plus any Steiner edges that replace them.
    steiner_points_rows = []   # points used per region (terminals + steiner)
    steiner_edges_rows  = []   # edges inside regions as coordinate segments

    fig = plt.figure(figsize=(7,7))

    # Determine which regions are “eligible” (>= 2 centralized terminals)
    # and keep masks for edge filtering and set-of-terminals accounting.
    cen_mask_all = nodes_df["centralized"].astype(bool).to_numpy()
    eligible_masks = []
    eligible_term_sets = []  # set of terminal indices used in each region tree
    for reg in regions_df.itertuples(index=False):
        cx, cy, r = float(reg.cx), float(reg.cy), float(reg.r)
        mask_inside = (xs - cx)**2 + (ys - cy)**2 <= (r**2)
        term_idx = np.where(mask_inside & cen_mask_all)[0]
        if term_idx.size >= 2:
            eligible_masks.append(mask_inside)
            eligible_term_sets.append(set(map(int, term_idx)))

    # Helper to decide if a kept baseline edge lies fully inside some eligible region.
    # If yes, we do not draw that baseline edge; it will be replaced by Steiner edges.
    def edge_entirely_inside_any_eligible_region(u: int, v: int) -> bool:
        for m in eligible_masks:
            if m[u] and m[v]:
                return True
        return False

    # Draw only the baseline edges that are not fully inside eligible regions.
    for _, erow in edges_df.iterrows():
        u, v = int(erow.u), int(erow.v)
        if edge_entirely_inside_any_eligible_region(u, v):
            continue
        x1, y1 = points[u]; x2, y2 = points[v]
        plt.plot([x1, x2], [y1, y2])

    # Scatter nodes (unchanged) so we can see centralized vs decentralized.
    cen_mask = nodes_df["centralized"].values.astype(bool)
    dec_mask = ~cen_mask
    plt.scatter(points[dec_mask,0], points[dec_mask,1], s=28, label="Decentralized")
    plt.scatter(points[cen_mask,0], points[cen_mask,1], s=40, label="Centralized")

    # For each region, draw the boundary and attempt Steiner rewiring.
    theta = np.linspace(0.0, 2.0*np.pi, 400)
    for reg in regions_df.itertuples(index=False):
        cx, cy, r = float(reg.cx), float(reg.cy), float(reg.r)
        plt.plot(cx + r*np.cos(theta), cy + r*np.sin(theta))

        dx = xs - cx
        dy = ys - cy
        inside_mask = (dx*dx + dy*dy) <= (r*r)
        term_idx = np.where(inside_mask & cen_mask)[0]
        if term_idx.size < 2:
            for nid in term_idx:
                steiner_points_rows.append({
                    "region_id": int(reg.region_id),
                    "kind": "terminal",
                    "orig_node_id": int(nid),
                    "x": float(xs[nid]),
                    "y": float(ys[nid])
                })
            continue

        terminals_xy = points[term_idx, :]  # (m,2)
        steiner_edges_coords, steiner_pts_xy = _run_steiner_via_mst_new(terminals_xy)

        for nid in term_idx:
            steiner_points_rows.append({
                "region_id": int(reg.region_id),
                "kind": "terminal",
                "orig_node_id": int(nid),
                "x": float(xs[nid]),
                "y": float(ys[nid])
            })
        for sp in steiner_pts_xy:
            steiner_points_rows.append({
                "region_id": int(reg.region_id),
                "kind": "steiner",
                "orig_node_id": -1,
                "x": float(sp[0]),
                "y": float(sp[1])
            })

        for (p1, p2) in steiner_edges_coords:
            steiner_edges_rows.append({
                "region_id": int(reg.region_id),
                "x1": float(p1[0]), "y1": float(p1[1]),
                "x2": float(p2[0]), "y2": float(p2[1])
            })
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    # Count how many terminals appear in any drawn tree (outside edges or region trees).
    terminals_from_outside_edges = {
        int(u) for (u, v, d) in kept if not edge_entirely_inside_any_eligible_region(u, v)
    } | {
        int(v) for (u, v, d) in kept if not edge_entirely_inside_any_eligible_region(u, v)
    }
    terminals_from_regions = set()
    for s in eligible_term_sets:
        terminals_from_regions |= s

    terminals_in_any_tree = terminals_from_outside_edges | terminals_from_regions
    num_terminals_in_any_tree = len(terminals_in_any_tree)

    # Finalize the figure with optional Steiner-point marks and a small legend note.
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    sx = [row["x"] for row in steiner_points_rows if row["kind"] == "steiner"]
    sy = [row["y"] for row in steiner_points_rows if row["kind"] == "steiner"]
    if len(sx) > 0:
        plt.scatter(sx, sy, s=80, marker="*", c="tab:red", label="Steiner", zorder=4)

    ax = plt.gca()
    ax.text(
        xmin + 0.02*(xmax - xmin),
        ymax - 0.02*(ymax - ymin),
        f"Terminals in trees: {num_terminals_in_any_tree} / {N}",
        ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85, lw=0.6)
    )

    plt.legend()
    plt.savefig("m1_plot_with_regions_steiner.png", dpi=160)
    plt.close(fig)


    # We write two CSVs: one listing terminals and any Steiner points,
    # and one listing all segments inside regions that were drawn.
    pd.DataFrame(steiner_points_rows, columns=["region_id","kind","orig_node_id","x","y"]) \
    .to_csv("region_steiner_points_all.csv", index=False)

    pd.DataFrame(steiner_edges_rows, columns=["region_id","x1","y1","x2","y2"]) \
    .to_csv("region_steiner_edges_all.csv", index=False)


    # We compare the baseline network against the regional-Steiner network.
    # Each connected centralized system pays one fixed substation cost.
    # We also (optionally) add fixed costs for any Steiner nodes inside regions.

    # Helper to test if a kept edge lies inside the same eligible region.
    # This is used to exclude such edges from the “outside” cost pile.
    def inside_same_eligible(u: int, v: int) -> bool:
        for m in eligible_masks:
            if m[u] and m[v]:
                return True
        return False

    # Baseline cost breakdown: line costs over kept edges and node costs by role.
    EGC_K = float(edges_df["egc"].sum()) if ("egc" in edges_df.columns) else 0.0
    mask_cen = nodes_df["centralized"].astype(bool).to_numpy()
    IGC_K = float(nodes_df.loc[mask_cen, "IGC"].sum())
    DSC_K = float(nodes_df.loc[~mask_cen, "DSC"].sum())
    TOTAL_K = EGC_K + IGC_K + DSC_K

    # Count baseline centralized components using DSU on kept edges.
    dsu_base = DSU(N)
    for (u, v, _) in kept:
        dsu_base.union(u, v)
    roots_base = {dsu_base.find(i) for i in range(N) if mask_cen[i]}
    num_substations_base = len(roots_base)

    # After-regions costs: keep outside edges; inside regions we use Steiner edges.
    EGC_outside = float(
        sum(UNIT_COST * float(np.hypot(points[u,0]-points[v,0], points[u,1]-points[v,1]))
            for (u, v, _) in kept if not inside_same_eligible(u, v))
    )
    EGC_steiner_inside = float(
        sum(UNIT_COST * math.hypot(er["x2"]-er["x1"], er["y2"]-er["y1"])
            for er in steiner_edges_rows)
    )

    # Recount components after regional merging: outside edges + union all terminals inside each region.
    dsu_after = DSU(N)
    for (u, v, _) in kept:
        if not inside_same_eligible(u, v):
            dsu_after.union(u, v)
    for reg in regions_df.itertuples(index=False):
        cx, cy, r = float(reg.cx), float(reg.cy), float(reg.r)
        mask_inside = (xs - cx)**2 + (ys - cy)**2 <= (r**2)
        term_idx = np.where(mask_inside & mask_cen)[0]
        if term_idx.size >= 2:
            base = int(term_idx[0])
            for t in term_idx[1:]:
                dsu_after.union(base, int(t))
    roots_after = {dsu_after.find(i) for i in range(N) if mask_cen[i]}
    num_substations_after = len(roots_after)

    # Debug print to show which path the Steiner adapter used for the last computed region.
    # This is diagnostic only and does not affect the cost math.
    used_path = "fallback"
    res = _call_user_mst_new(points[term_idx, :])
    if res is not None:
        used_path = "MST_NEW"
    else:
        used_path = "fallback"
    print(f"[Region {int(reg.region_id)}] terminals={term_idx.size}, path={used_path}")

    # Sample a reproducible per-substation cost and compute totals by scenario.
    SUB_COST_K   = num_substations_base  * SUBSTATION_COST
    SUB_COST_EST = num_substations_after * SUBSTATION_COST

    # If Steiner fixed costs were not computed earlier, do it now and save.
    try:
        ST_COST_SUM
    except NameError:
        rng = np.random.default_rng(SEED + 4242)
        unique_steiner_keys = {
            (int(r["region_id"]), round(float(r["x"]), 12), round(float(r["y"]), 12))
            for r in steiner_points_rows if r["kind"] == "steiner"
        }
        st_cost_map = {key: float(rng.uniform(*ST_COST_RANGE)) for key in unique_steiner_keys}
        ST_COST_SUM = float(sum(st_cost_map.values()))
        pd.DataFrame(
            [{"region_id": rid, "x": x, "y": y, "st_cost": c}
            for (rid, x, y), c in st_cost_map.items()]
        ).to_csv("steiner_costs.csv", index=False)

    IGC_EST, DSC_EST = IGC_K, DSC_K
    EGC_EST = EGC_outside + EGC_steiner_inside

    TOTAL_K_with_SUB   = TOTAL_K + SUB_COST_K
    TOTAL_EST_with_SUB = (EGC_EST + IGC_EST + DSC_EST + ST_COST_SUM) + SUB_COST_EST

    # The table shows line-only EGC, node costs, Steiner fixed costs,
    # the number of substations, and the grand total with substations.
    diff = TOTAL_EST_with_SUB - TOTAL_K_with_SUB
    print(diff, "components: ", num_substations_base)
    if(diff < 0): 
        print(SEED)
    compare_df = pd.DataFrame([
        {"tree": "Kruskal (TK)",
        "substations": num_substations_base,
        "SUB_cost_each": SUBSTATION_COST,
        "SUB_cost_total": SUB_COST_K,
        "edges": int(len(kept)),
        "EGC_line": EGC_K,
        "IGC": IGC_K, "DSC": DSC_K,
        "ST_cost": 0.0,
        "TOTAL": TOTAL_K_with_SUB},
        {"tree": "With Steiner in regions (TEST)",
        "substations": num_substations_after,
        "SUB_cost_each": SUBSTATION_COST,
        "SUB_cost_total": SUB_COST_EST,
        "edges": int(len(steiner_edges_rows) +
                    sum(1 for (u, v, _) in kept if not inside_same_eligible(u, v))),
        "EGC_line": EGC_EST,
        "IGC": IGC_EST, "DSC": DSC_EST,
        "ST_cost": ST_COST_SUM,
        "TOTAL": TOTAL_EST_with_SUB}
    ])
    compare_df.to_csv("tree_costs_comparison.csv", index=False)

# for i in range(1722, 1000000):
#     fun(i)

fun(347)