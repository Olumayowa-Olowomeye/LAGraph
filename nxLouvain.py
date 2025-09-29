import time
import numpy as np
from scipy.io import mmread
import networkx as nx
import igraph as ig
# import community as community_louvain  # for networkx Louvain
import os

# List of input graphs in Matrix Market format
tests = [
    "20000node.mtx",
    "50000node.mtx",
    "500000node.mtx",
    "2000000node.mtx",
    "com-Amazon.mtx"
]

# Number of trials for timing
NUM_TRIALS = 10
THREAD_COUNTS = range(1, 24)

def load_graph_networkx(file_path):
    matrix = mmread(file_path).tocoo()
    G = nx.Graph()
    for u, v in zip(matrix.row, matrix.col):
        if u != v:
            G.add_edge(int(u), int(v))
    return G

def load_graph_igraph(file_path):
    matrix = mmread(file_path).tocoo()
    edges = list(zip(map(int, matrix.row), map(int, matrix.col)))
    g = ig.Graph(edges=edges, directed=False)
    g.simplify()
    return g

def run_louvain_networkx(G):
    comms = nx.algorithms.community.louvain_communities(G, seed=42)  # Add seed for consistency
    modularity = nx.algorithms.community.modularity(G, comms)
    return modularity


def run_louvain_igraph(g):
    partition = g.community_multilevel()
    return partition.modularity

def benchmark():
    results = []

    for graph_file in tests:
        print(f"\nProcessing: {graph_file}")
        nx_graph = load_graph_networkx(graph_file)
        ig_graph = load_graph_igraph(graph_file)

        for threads in THREAD_COUNTS:
            print(f"  Threads: {threads}")

            # --- iGraph ---
            os.environ["OMP_NUM_THREADS"] = str(threads)
            ig_times = []
            ig_mod = None
            for _ in range(NUM_TRIALS):
                start = time.time()
                mod = run_louvain_igraph(ig_graph)
                elapsed = time.time() - start
                ig_times.append(elapsed)
                ig_mod = mod
            avg_ig_time = np.mean(ig_times)

            # --- NetworkX ---
            # Note: networkx doesn't support threading in Louvain (pure Python)
            nx_times = []
            nx_mod = None
            for _ in range(NUM_TRIALS):
                start = time.time()
                mod = run_louvain_networkx(nx_graph)
                elapsed = time.time() - start
                nx_times.append(elapsed)
                nx_mod = mod
            avg_nx_time = np.mean(nx_times)

            results.append({
                'graph': graph_file,
                'threads': threads,
                'nx_avg_time': avg_nx_time,
                'nx_modularity': nx_mod,
                'ig_avg_time': avg_ig_time,
                'ig_modularity': ig_mod
            })

    return results

if __name__ == "__main__":
    import csv

    data = benchmark()

    # Save to CSV
    with open("louvain_benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'graph', 'threads',
            'nx_avg_time', 'nx_modularity',
            'ig_avg_time', 'ig_modularity'
        ])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print("\nBenchmark complete. Results saved to louvain_benchmark_results.csv.")
