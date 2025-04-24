from itertools import combinations
import igraph as ig
import time
import gzip
import pickle
import pathlib
from loguru import logger


from . import PlaceDB, DesignReader, SoftMacroReader

def build_soft_macro_placedb(reader: DesignReader, coverage: float = 0.95, gamma: float = 1.1, grid:int=224, cache_root=None) -> PlaceDB:
    """
    Build a soft macro placedb from a place db.
    """
    placedb = PlaceDB(reader)
    graph = placedb_to_graph(placedb)
    canvas_width, canvas_height = reader.canvas_size
    base_area = canvas_width / grid * canvas_height / grid

    subgraph_list = graph_partition(graph, cache_tag=placedb.cache_tag, coverage=coverage, cache_root=cache_root)

    total_nodes, total_edges = 0, 0
    for i, subgraph in enumerate(subgraph_list):
        area = sum(subgraph.vs['area'])
        logger.info(f"Subgraph {i}: nodes = {len(subgraph.vs)} edges = {len(subgraph.es)} area = {area:.3e} area ratio: {area/base_area:.3f}")
        total_nodes += len(subgraph.vs)
        total_edges += len(subgraph.es)

    return PlaceDB(SoftMacroReader(reader, subgraph_list, gamma))


def placedb_to_graph(placedb: PlaceDB) -> ig.Graph:
    edges = []
    node_attrs = {'area': [], 'name': []}
    edge_attrs = {'weight': []}
    node_name2id = {}
    empty_nets, pin_nets = 0, 0
    for net_name in placedb.net_info:
        source_info = placedb.net_info[net_name]["source"]
        if "node_type" not in source_info:
            # logger.info(f"{net_name=} is empty, skip")
            empty_nets += 1
            continue
        if source_info["node_type"] == "PIN":
            # logger.info(f"{net_name=} Input is PIN, skip")
            pin_nets += 1
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

            if node_name1 not in node_name2id:
                node_info = placedb.cell_info[node_name1]
                width = node_info["width"]
                height = node_info["height"]
                node_name2id[node_name1] = len(node_name2id)
                node_attrs["area"].append(width * height)
                node_attrs["name"].append(node_name1)
                assert len(node_name2id) == len(node_attrs["area"]) == len(node_attrs["name"])

            if node_name2 not in node_name2id:
                node_info = placedb.cell_info[node_name2]
                width = node_info["width"]
                height = node_info["height"]
                node_name2id[node_name2] = len(node_name2id)
                node_attrs["area"].append(width * height)
                node_attrs["name"].append(node_name2)
                assert len(node_name2id) == len(node_attrs["area"]) == len(node_attrs["name"])

            pins1 = node1["pins"].keys()
            pins2 = node2["pins"].keys()
            weight = len(pins1) * len(pins2)
            edges.append((node_name2id[node_name1], node_name2id[node_name2]))
            edge_attrs['weight'].append(weight)
            assert len(edges) == len(edge_attrs["weight"])
    logger.info(f"raw nets: {len(placedb.net_info)}, remove nets: {empty_nets+pin_nets} = (empty:{empty_nets}, pin:{pin_nets})")
    G = ig.Graph(len(node_name2id), edges, vertex_attrs=node_attrs, edge_attrs=edge_attrs)
    logger.info(f"stdcell in placedb: {len(placedb.cell_info)}, stdcell in graph: {len(G.vs)}")
    return G


def graph_partition(
    G: ig.Graph,
    cache_tag: str,
    coverage: float,
    cache_root: str = None,
) -> list[ig.Graph]:
    graph_partition_pkl = None
    if cache_root is not None:
        graph_partition_pkl = pathlib.Path(cache_root) / f"{cache_tag}_gp.pkl.gz"


    if graph_partition_pkl is not None and graph_partition_pkl.exists():
        with gzip.open(graph_partition_pkl, "rb") as f:
            subgraphs, discards = pickle.load(f)
        logger.info(f"read graph partition result from {graph_partition_pkl}")
    else:
        subgraphs, discards = graph_louvain(G, coverage)
        if graph_partition_pkl is not None:
            with gzip.open(graph_partition_pkl, "wb") as f:
                pickle.dump((subgraphs, discards), f)
            logger.info(f"save graph partition result into {graph_partition_pkl}")

    original_area = sum(G.vs['area'])
    logger.info(f"Original# node: {len(G.vs):.3e}, area:{original_area:.3e}, edge: {len(G.es):.3e}")

    subgraph_node, subgraph_edge, subgraph_area = 0, 0, 0
    for graph in subgraphs:
        subgraph_node += len(graph.vs)
        subgraph_edge += len(graph.es)
        subgraph_area += sum(graph.vs['area'])

    logger.info(f"Subgraph# node: {subgraph_node:.3e}, area:{subgraph_area:.3e}, edge: {subgraph_edge:.3e}")

    discard_node, discard_edge, discard_area = 0, 0, 0
    for graph in discards:
        discard_node += len(graph.vs)
        discard_edge += len(graph.es)
        discard_area += sum(graph.vs['area'])
    logger.info(f"Discards# node: {discard_node:.3e}, area:{discard_area:.3e}, edge: {discard_edge:.3e}")

    return subgraphs

def graph_louvain(G: ig.Graph, coverage:float=0.95) -> tuple[list[ig.Graph], list[ig.Graph]]:

    assert isinstance(G, ig.Graph)
    assert 'weight' in G.es.attribute_names()
    assert 'area' in G.vs.attribute_names()
    assert 0 < coverage <= 1

    t0 = time.time()
    communities = G.community_multilevel(weights='weight', resolution=1)
    logger.info(f"Time for louvain partition: {time.time() - t0:.2f}s")

    soft_macro = {}
    total_area, total_node, total_edge = 0, 0, 0
    for i, community in enumerate(communities):

        subgraph = G.subgraph(community)
        area = sum(subgraph.vs['area'])
        node = len(subgraph.vs)
        edge = len(subgraph.es)

        soft_macro[i] = {
            'graph': subgraph,
            'area': area,
            'node': node,
            'edge': edge
        }
        total_area += area
        total_node += node
        total_edge += edge

    subgraphs, discards = [], []
    cum_area, cum_node, cum_edge = 0, 0, 0
    area_threshold = total_area * coverage
    node_threshold = total_node * coverage
    edge_threshold = total_edge * coverage
    for _, info in sorted(soft_macro.items(), key=lambda x:x[1]['area'], reverse=True):
        area = info['area']
        node = info['node']
        edge = info['edge']
        cum_area += area
        cum_node += node
        cum_edge += edge

        if cum_area < area_threshold or cum_node < node_threshold or cum_edge < edge_threshold:
            subgraphs.append(info['graph'])
        else:
            discards.append(info['graph'])

    return subgraphs, discards
    