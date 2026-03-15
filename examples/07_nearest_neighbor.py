"""Nearest Neighbor Search -- CPU KDTree vs GPU Spatial Hash."""
import time
import numpy as np, mlx.core as mx
from open3d_mlx.ops import NearestNeighborSearch, FixedRadiusIndex

rng = np.random.default_rng(42)
N = 10000
points = mx.array(rng.standard_normal((N, 3)).astype(np.float32))
queries = mx.array(rng.standard_normal((1000, 3)).astype(np.float32))

# CPU KDTree
t0 = time.perf_counter()
nns = NearestNeighborSearch(points)
indices_knn, dists_knn = nns.knn_search(queries, k=10)
t_knn = time.perf_counter() - t0
print(f"CPU KDTree (k=10): {t_knn * 1000:.1f}ms for {queries.shape[0]} queries")

# GPU Spatial Hash
t0 = time.perf_counter()
fri = FixedRadiusIndex(points, radius=0.5)
indices_fr, dists_fr = fri.search_nearest(queries)
t_fr = time.perf_counter() - t0
print(f"GPU Spatial Hash:  {t_fr * 1000:.1f}ms for {queries.shape[0]} queries")

# Hybrid search
indices_h, dists_h, counts = nns.hybrid_search(queries, radius=0.5, max_nn=10)
avg_neighbors = float(np.array(counts).mean())
print(f"\nHybrid search (r=0.5, max_nn=10): avg {avg_neighbors:.1f} neighbors/point")

print("\nExample 07 complete")
