import networkx as nx
import pandas as pd
from networkx.algorithms import community
import glob
import warnings
import networkx as nx
import pandas as pd
from networkx.algorithms import community as nx_comm
import os
import matplotlib.pyplot as plt
import numpy as np
import datashader.bundling as bd
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


def compute_network_stats(G):
    # ensure undirected
    G = G.to_undirected()
    N = G.number_of_nodes()
    M = G.number_of_edges()

    # connected component metrics
    comps = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in comps]
    largest = max(comps, key=len)
    Gc = G.subgraph(largest)
    
    # ── community detection & modularity ─────────────────────────
    # use greedy modularity communities
    comms = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
    num_com = len(comms)
    mod     = nx_comm.modularity(G, comms, weight='weight')

    stats = {
        'nodes': N,
        'edges': M,
        'density': nx.density(G),
        'components': len(comps),
        'largest_comp_size': len(largest),
        'diameter': nx.diameter(Gc),
        'avg_shortest_path': nx.average_shortest_path_length(Gc),
        'transitivity': nx.transitivity(G),
        'avg_clustering': nx.average_clustering(G),
        'mean_degree': sum(dict(G.degree()).values())/N,
        'degree_heterogeneity': (pd.Series(dict(G.degree())).var() /
                                 pd.Series(dict(G.degree())).mean()),
        'degree_assortativity': nx.degree_assortativity_coefficient(G),
        'num_communities': num_com,
        'modularity': mod,
    }
    
    return stats


def compute_node_distributions(G):
    Gu = G.to_undirected()

    # 1) Degree, betweenness, closeness
    deg = dict(Gu.degree())
    btw = nx.betweenness_centrality(Gu)
    clo = nx.closeness_centrality(Gu)

    # 2) Eigenvector per component, with special‐case for size<3
    eig = {}
    for comp in nx.connected_components(Gu):
        sub = Gu.subgraph(comp)
        if len(sub) < 3:
            # two‐node or single‐node: give them equal centrality
            for node in sub.nodes():
                eig[node] = 1.0
        else:
            sub_eig = nx.eigenvector_centrality_numpy(sub)
            eig.update(sub_eig)
    
    # 3) PageRank with fallback to the eigenvector proxy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pr = nx.pagerank(Gu, alpha=0.85, max_iter=1000, tol=1e-06)
    except Exception:
        warnings.warn("PageRank failed; using eigenvector as proxy")
        pr = eig

    # 4) Pull out ASV_ID if present
    ids = [Gu.nodes[n].get('ASV_ID', n) for n in Gu.nodes()]

    # 5) Build DataFrame
    df = pd.DataFrame({
        'node':        list(deg.keys()),
        'ASV_ID':      ids,
        'degree':      list(deg.values()),
        'betweenness': [btw[n] for n in deg],
        'closeness':   [clo[n] for n in deg],
        'eigenvector': [eig[n] for n in deg],
        'pagerank':    [pr[n]  for n in deg]
    })
    return df

all_stats = []
all_nodes  = []

plot_dir = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/plots"
os.makedirs(plot_dir, exist_ok=True)

for path in glob.glob("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_networks/*.graphml"):
    name = path.split("/")[-1].replace(".graphml","")
    pt = os.path.splitext(os.path.basename(path))[0]

    print(f"Processing {name}...")
    print(f"Bundling via GraphViz: {pt}")

    # 1) load with NetworkX
    G = nx.read_graphml(path)

    # 2) convert to AGraph
    A = to_agraph(G)

    # 3) tell GraphViz to bundle edges                    
    A.graph_attr.update(concentrate="true", splines="curved")
    A.node_attr.update(shape="circle", style="filled", fillcolor="lightgray", fontsize="8")
    A.edge_attr.update(color="gray", penwidth="0.5", alpha="0.6")

    # 4) layout & render
    A.layout(prog="neato")  # or "fdp" / "sfdp" for large graphs
    out_svg = os.path.join(plot_dir, f"{pt}_bundled.pdf")
    A.draw(out_svg)
    print(" → wrote", out_svg)

    gs = compute_network_stats(G)
    gs['network'] = name
    all_stats.append(gs)

    nd = compute_node_distributions(G)
    nd['network'] = name
    all_nodes.append(nd)

# compile into tables
stats_df = pd.DataFrame(all_stats).set_index('network')
nodes_df = pd.concat(all_nodes, ignore_index=True)

# save or inspect
stats_df.to_csv("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/network_global_stats.tsv", sep="\t")
nodes_df.to_csv("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/network_node_stats.tsv", sep="\t")

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# — 1) Build a merged ASV–ASV graph for layout ————————————————
G_asv = nx.Graph()
for path in glob.glob("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_networks/*.graphml"):
    g = nx.read_graphml(path)
    for u, v, data in g.edges(data=True):
        w = float(data.get("weight", 1.0))
        if G_asv.has_edge(u, v):
            G_asv[u][v]["weight"] += w
        else:
            G_asv.add_edge(u, v, weight=w)

# 2) Compute an ASV spring layout (positions for ASV nodes)
pos = nx.spring_layout(G_asv, weight="weight", seed=42)

# — 3) Build meta‐graph with ASVs + patients ——————————————————
G_meta = nx.Graph()

# 3a) Add ASV nodes (carry over whatever attributes you like)
for node, data in G_asv.nodes(data=True):
    G_meta.add_node(node,
                    node_type="asv",
                    **{k: data[k] for k in data if k != "weight"})

# 3b) Add patient nodes with attributes from stats_df
for pt, row in stats_df.iterrows():
    G_meta.add_node(pt, node_type="patient",
                    mean_degree = float(row["mean_degree"]),
                    size        = float(row["mean_degree"])*200)

# 3c) Add patient–patient edges based on global‐stat correlations
#    threshold to only keep strong similarities
num_cols = stats_df.select_dtypes(include=[float, int])
corr = num_cols.T.corr()
sim_thresh = 0.5  # tune this
for i, j in combinations(stats_df.index, 2):
    w = corr.at[i, j]
    if w >= sim_thresh:
        G_meta.add_edge(i, j,
                        edge_type="patient_similarity",
                        weight = w)

# 3d) (Optional) Add membership edges if you want them drawn
for path in glob.glob("/home/ryan/.../patient_networks/*.graphml"):
    pt = os.path.basename(path).replace('.graphml','')
    g = nx.read_graphml(path)
    for asv in g.nodes():
        if G_meta.has_node(asv):
            # binary membership
            G_meta.add_edge(pt, asv,
                            edge_type="membership",
                            weight=1.0)

# — 4) Compute patient‐node positions as centroids of their ASVs ————
for pt in stats_df.index:
    asvs = [nbr for nbr in G_meta.neighbors(pt)
            if G_meta.nodes[nbr]['node_type']=='asv']
    if asvs:
        coords = np.array([pos[a] for a in asvs])
        pos[pt] = coords.mean(axis=0)
    else:
        pos[pt] = np.random.RandomState(42).rand(2)  # isolated patient

# — 5) Draw the combined plot —————————————————————————————
plt.figure(figsize=(12,12))

# draw ASV nodes
asv_nodes = [n for n,d in G_meta.nodes(data=True) if d['node_type']=='asv']
nx.draw_networkx_nodes(G_meta, pos,
                       nodelist=asv_nodes,
                       node_size=20,
                       node_color="lightgray",
                       alpha=0.6)

# draw patient nodes
patient_nodes = [n for n,d in G_meta.nodes(data=True) if d['node_type']=='patient']
sizes = [G_meta.nodes[n]['size'] for n in patient_nodes]
nx.draw_networkx_nodes(G_meta, pos,
                       nodelist=patient_nodes,
                       node_size=sizes,
                       node_color="red",
                       edgecolors="black",
                       linewidths=1.0)

# draw only patient–patient edges
patient_edges = [(u,v) for u,v,d in G_meta.edges(data=True)
                 if d['edge_type']=="patient_similarity"]
nx.draw_networkx_edges(G_meta, pos,
                       edgelist=patient_edges,
                       width=[G_meta[u][v]['weight']*2 for u,v in patient_edges],
                       edge_color="blue",
                       alpha=0.8)

# label patients
labels = {n:n for n in patient_nodes}
nx.draw_networkx_labels(G_meta, pos,
                        labels=labels,
                        font_size=14,
                        font_weight="bold")

plt.axis("off")
plt.tight_layout()
plt.savefig("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_metagraph.svg", bbox_inches='tight')
plt.savefig("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_metagraph.pdf", bbox_inches='tight')
nx.write_graphml(G_meta, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_metagraph.graphml")
