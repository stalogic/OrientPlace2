from itertools import combinations
from scipy.stats import gmean
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
import networkx as nx
import numpy as np
import time
import gzip
import pickle
import pathlib


from . import PlaceDB, DesignReader, SoftMacroReader

def build_soft_macro_placedb(reader: DesignReader, gamma: float = 1.1) -> PlaceDB:
    """
    Build a soft macro placedb from a place db.
    """
    placedb = PlaceDB(reader)
    graph = placedb_to_graph(placedb)
    base_area = gmean([macro["width"] * macro["height"] for macro in placedb.hard_macro_info.values()])

    start_time = time.time()
    subgraph_list = graph_partition(graph, base_area, cache_tag=placedb.cache_tag)

    total_nodes, total_edges = 0, 0
    for i, subgraph in enumerate(subgraph_list):
        area = sum([attr["area"] for _, attr in subgraph.nodes(data=True)])
        print(f"Subgraph {i}: nodes = {len(subgraph.nodes())} edges = {len(subgraph.edges())} area = {area:.3e} area ratio: {area/base_area:.3f}")
        total_nodes += len(subgraph.nodes())
        total_edges += len(subgraph.edges())
    
    print(f"Time used for graph cluster: {time.time() - start_time:.1f}")

    return PlaceDB(SoftMacroReader(reader, subgraph_list, gamma))


def placedb_to_graph(placedb: PlaceDB) -> nx.Graph:
    G = nx.Graph()

    for net_name in placedb.net_info:
        source_info = placedb.net_info[net_name]["source"]
        if "node_type" not in source_info:
            print(f"{net_name=} is empty, skip")
            continue
        if source_info["node_type"] == "PIN":
            print(f"{net_name=} Input is PIN, skip")
            continue

        net_nodes = placedb.net_info[net_name]["nodes"].values()
        for node1, node2 in combinations(net_nodes, 2):
            node_name1 = node1["key"]
            node_name2 = node2["key"]

            # 过滤掉net中的macro
            if node_name1 in placedb.macro_info or node_name2 in placedb.macro_info:
                continue
            
            # net的中的port，不应该出现在nodes中
            if node_name1 in placedb.port_info or node_name2 in placedb.port_info:
                assert False

            if node_name1 not in G:
                node_info = placedb.cell_info[node_name1]
                width = node_info["width"]
                height = node_info["height"]
                G.add_node(node_name1, area=width * height)

            if node_name2 not in G:
                node_info = placedb.cell_info[node_name2]
                width = node_info["width"]
                height = node_info["height"]
                G.add_node(node_name2, area=width * height)

            pins1 = node1["pins"].keys()
            pins2 = node2["pins"].keys()
            for pin1 in pins1:
                for pin2 in pins2:
                    G.add_edge(node_name1, node_name2, net_name=net_name, pin1=pin1, pin2=pin2)

    return G

def graph_partition(
    G: nx.Graph,
    base_area: float,
    cache_tag: str,
    min_ratio=0.1,
    max_ratio=1.5,
    cache_root: str = "./cache",
) -> list[nx.Graph]:

    cache_path = pathlib.Path(cache_root)
    graph_partition_result = cache_path / f"{cache_tag}_gp.pkl.gz"
    print(f"graph partition cache: {graph_partition_result.name}")

    if graph_partition_result.exists():
        with gzip.open(graph_partition_result, "rb") as f:
            subgraphs, discards = pickle.load(f)
        print(f"read graph partition result from {graph_partition_result}")
    else:
        # 使用谱聚类算法
        sc = SpectralClustering(2, affinity="precomputed", random_state=0)
        print(f"{' start graph partition ':#^80}")
        subgraphs, discards = clustering_and_partition(sc, G, base_area, min_ratio, max_ratio)
        print(f"{' finish graph partition ':#^80}")
        with gzip.open(graph_partition_result, "wb") as f:
            pickle.dump((subgraphs, discards), f)
        print(f"save graph partition result into {graph_partition_result}")

    original_area = sum([node["area"] for _, node in G.nodes(data=True)])
    print(f"Original# node: {len(G.nodes):.3e}, area:{original_area:.3e}, edge: {len(G.edges):.3e}")

    subgraph_node, subgraph_edge, subgraph_area = 0, 0, 0
    for graph in subgraphs:
        subgraph_node += len(graph.nodes)
        subgraph_edge += len(graph.edges)
        subgraph_area += sum([node["area"] for _, node in graph.nodes(data=True)])

    print(f"Subgraph# node: {subgraph_node:.3e}, area:{subgraph_area:.3e}, edge: {subgraph_edge:.3e}")

    discard_node, discard_edge, discard_area = 0, 0, 0
    for graph in discards:
        discard_node += len(graph.nodes)
        discard_edge += len(graph.edges)
        discard_area += sum([node["area"] for _, node in graph.nodes(data=True)])
    print(f"Discards# node: {discard_node:.3e}, area:{discard_area:.3e}, edge: {discard_edge:.3e}")

    return subgraphs

def clustering_and_partition(sc: SpectralClustering, G: nx.Graph, base_area: float, min_ratio=0.1, max_ratio=1.5) -> tuple[list[nx.Graph], list[nx.Graph]]:

    # 获取图的邻接矩阵
    adj_matrix: np.ndarray = nx.to_numpy_array(G, dtype="bool")
    sprase_matrix = csr_matrix(adj_matrix)
    labels = sc.fit_predict(sprase_matrix)
    subgraphs = []
    discards = []
    num_clusters = 2
    for i in range(num_clusters):
        subgraph_nodes = [node for node, label in zip(G.nodes(), labels) if label == i]
        subgraph: nx.Graph = G.subgraph(subgraph_nodes)
        subgraph_area = sum([node["area"] for _, node in subgraph.nodes(data=True)])
        print(f"sub graph# node:{len(subgraph.nodes)}, edge: {len(subgraph.edges)}, area: {subgraph_area:.3e}")

        if subgraph_area > max_ratio * base_area:
            graphs, discard_graphs = clustering_and_partition(sc, subgraph, base_area, min_ratio, max_ratio)
            subgraphs.extend(graphs)
            discards.extend(discard_graphs)
        elif min_ratio is not None and subgraph_area < min_ratio * base_area:
            discards.append(subgraph)
        else:
            subgraphs.append(subgraph)
    return subgraphs, discards
    