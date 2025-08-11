import networkx as nx
import numpy as np
import pandas as pd


def calculate_link_times(graph):
    result = {}
    for u, v, d in graph.edges(data=True):
        ts = d.get('timestamp')
        if ts is not None:
            if isinstance(ts, pd.Timedelta):
                raise ValueError("Timestamp is a Timedelta, which is not expected.")
            ts = pd.to_datetime(ts).timestamp()
            result[(u, v)] = float(ts)
    return result


def calculate_graph_period(link_times):
    timestamps = list(link_times.values())
    return max(timestamps) - min(timestamps)


def exponential_decay(T0, t, t_ab):
    for var in ['t', 't_ab']:
        val = locals()[var]
        if isinstance(val, pd.Timestamp):
            locals()[var] = val.timestamp()
        elif isinstance(val, pd.Timedelta):
            locals()[var] = val.total_seconds()
    delta = float(t - t_ab)
    return np.exp((-np.log(2) / T0) * delta)


def calculate_node_sets(graph):
    return {node: set(graph.neighbors(node)) for node in graph.nodes()}


def extract_length_3_paths_after_edge_removal(graph):
    paths = {}
    for u, i in graph.edges():
        g_copy = graph.copy()
        g_copy.remove_edge(u, i)
        paths[(u, i)] = [p for p in nx.all_simple_paths(g_copy, source=u, target=i, cutoff=3) if len(p) == 4]
    return paths


def count_length_3_paths(paths):
    """
    Count the number of length-3 paths for each pair (u, i) present in the paths dictionary.

    Args:
    - paths (dict): Dictionary mapping each edge to the list of length-3 paths after edge removal.

    Returns:
    - paths_counts (dict): Dictionary mapping each pair (u, i) to the number of length-3 paths.
    """
    paths_counts = {}

    for edge, length_3_paths in paths.items():
        u, i = edge
        num_paths = len(length_3_paths)
        paths_counts[(u, i)] = num_paths

    return paths_counts


def calculate_duration_of_length_3_paths(graph):
    paths = extract_length_3_paths_after_edge_removal(graph)
    link_times = calculate_link_times(graph)
    durations = {}
    for u, v in graph.edges():
        l3_paths = paths.get((u, v), []) + paths.get((v, u), [])
        if not l3_paths:
            durations[(u, v)] = 0
            continue
        timestamps = []
        for path in l3_paths:
            for i in range(3):
                ts = link_times.get((path[i], path[i+1]), link_times.get((path[i+1], path[i])))
                if ts is not None:
                    timestamps.append(ts)
        durations[(u, v)] = max(timestamps) - min(timestamps) if timestamps else 0
    return durations


def calculate_age_of_length_3_paths(graph):
    link_times = calculate_link_times(graph)
    latest_t = max(link_times.values())
    paths = extract_length_3_paths_after_edge_removal(graph)
    ages = {}
    for u, v in graph.edges():
        l3_paths = paths.get((u, v), []) + paths.get((v, u), [])
        ts_list = [link_times.get((a, b), link_times.get((b, a), 0.0)) for path in l3_paths for a, b in zip(path, path[1:])]
        latest_path_t = max(ts_list) if ts_list else 0.0
        ages[(u, v)] = latest_t - latest_path_t
    return ages


def calculate_CN_l3_similarity(graph):
    node_sets = calculate_node_sets(graph)
    CN = {}
    for u, i in graph.edges():
        iu, ui = node_sets[u], node_sets[i]
        CN[(u, i)] = sum(len(iu & node_sets[u_prime]) for u_prime in ui if u_prime != u) + \
                     sum(len(ui & node_sets[i_prime]) for i_prime in iu if i_prime != i)
    return CN


def calculate_jaccard_l3_similarity(graph):
    node_sets = calculate_node_sets(graph)
    jaccard = {}
    for u, i in graph.edges():
        iu, ui = node_sets[u], node_sets[i]
        jaccard_u = sum(len(iu & node_sets[u_prime]) / len(iu | node_sets[u_prime]) for u_prime in ui if u_prime != u and len(iu | node_sets[u_prime]) > 0)
        jaccard_i = sum(len(ui & node_sets[i_prime]) / len(ui | node_sets[i_prime]) for i_prime in iu if i_prime != i and len(ui | node_sets[i_prime]) > 0)
        jaccard[(u, i)] = jaccard_u + jaccard_i
    return jaccard


def calculate_CN_l3_dynamique(graph, T0):
    node_sets = calculate_node_sets(graph)
    link_times = calculate_link_times(graph)
    t = max(link_times.values())
    CN_dyn = {}
    for u, i in graph.edges():
        iu, ui = node_sets[u], node_sets[i]
        numerator = 0
        for u_prime in ui:
            for i_prime in iu & node_sets.get(u_prime, set()):
                t_ui = link_times.get((u, i_prime), link_times.get((i_prime, u)))
                t_upi = link_times.get((u_prime, i), link_times.get((i, u_prime)))
                if t_ui is not None and t_upi is not None:
                    numerator += exponential_decay(0.5 * T0, t, abs(t_ui - t_upi))
        CN_dyn[(u, i)] = numerator
    return CN_dyn


def calculate_jaccard_l3_dynamique(graph, T0):
    node_sets = calculate_node_sets(graph)
    link_times = calculate_link_times(graph)
    t = max(link_times.values())
    j_l3_dyn = {}
    for u, i in graph.edges():
        iu, ui = node_sets[u], node_sets[i]
        numerator = denom_shared = denom_residual = 0
        for u_prime in ui:
            iu_prime = node_sets.get(u_prime, set())
            shared = iu & iu_prime
            for i_prime in shared:
                t_ui = link_times.get((u, i_prime), link_times.get((i_prime, u)))
                t_upi = link_times.get((u_prime, i), link_times.get((i, u_prime)))
                if t_ui and t_upi:
                    val = exponential_decay(0.5 * T0, t, abs(t_ui - t_upi))
                    numerator += val
                    denom_shared += val
        for ip in iu:
            ts = link_times.get((u, ip), link_times.get((ip, u)))
            if ts:
                denom_residual += exponential_decay(0.5 * T0, t, abs(t - ts))
        for ip in iu_prime:
            ts = link_times.get((u_prime, ip), link_times.get((ip, u_prime)))
            if ts:
                denom_residual += exponential_decay(0.5 * T0, t, abs(t - ts))
        denom_total = denom_shared + denom_residual
        j_l3_dyn[(u, i)] = numerator / denom_total if denom_total > 0 else 0
    return j_l3_dyn


def calculate_katz_l3_dynamique(graph, T0):
    link_times = calculate_link_times(graph)
    paths = extract_length_3_paths_after_edge_removal(graph)
    katz = {}
    t = max(link_times.values())
    for edge in graph.edges():
        total_weight = 0
        for path in paths.get(edge, []) + paths.get((edge[1], edge[0]), []):
            weight = 0
            for a, b in zip(path, path[1:]):
                tab = link_times.get((a, b), link_times.get((b, a)))
                if tab is not None:
                    weight += exponential_decay(T0, t, tab)
            total_weight += weight
        katz[edge] = total_weight
    return katz


def last_t(graph, T0):
    tu, ti = calculate_last_interactions(graph)
    link_times = calculate_link_times(graph)
    t = max(link_times.values())
    last_tu = {u: exponential_decay(T0, t, ts) for u, ts in tu.items()}
    last_ti = {i: exponential_decay(T0, t, ts) for i, ts in ti.items()}
    return last_tu, last_ti


def calculate_last_interactions(graph):
    tu = {}
    ti = {}
    for u, i, data in graph.edges(data=True):
        ts = data.get("timestamp")
        if ts is not None:
            ts = pd.to_datetime(ts).timestamp()
            tu[u] = max(tu.get(u, 0), ts)
            ti[i] = max(ti.get(i, 0), ts)
    return tu, ti

def calculate_last_tu_ti_diff(graph, T0):
    tu, ti = calculate_last_interactions(graph)
    return {
        (u, i): exponential_decay(T0, tu[u], ti[i]) if u in tu and i in ti else 0
        for u, i in graph.edges()
    }



def calculate_category_features(graph):
    # Ajout descripteurs content_r_ui_mean et content_r_ui_std
    content_r_ui_mean = {}
    content_r_ui_std = {}

    for u, i, data in graph.edges(data=True):
        # Récupération des genres de l'item i
        genres_i = {k for k, v in graph.nodes[i].items() if k.startswith('cat_') and v == 1}

        if not genres_i:
            content_r_ui_mean[(u, i)] = 0
            content_r_ui_std[(u, i)] = 0
            continue

        # Items voisins de u (c'est-à-dire notés par u)
        rated_items_by_u = set(graph[u])

        # On exclut i et on filtre les items ayant un genre commun
        items_same_genre = []
        for j in rated_items_by_u:
            if j == i:
                continue
            genres_j = {k for k, v in graph.nodes[j].items() if k.startswith('cat_') and v == 1}
            if genres_i & genres_j:
                rating = graph[u][j].get("rating")
                if rating is not None:
                    items_same_genre.append(rating)

        if items_same_genre:
            content_r_ui_mean[(u, i)] = np.mean(items_same_genre)
            content_r_ui_std[(u, i)] = np.std(items_same_genre)
        else:
            content_r_ui_mean[(u, i)] = 0
            content_r_ui_std[(u, i)] = 0
            
    return content_r_ui_mean, content_r_ui_std



def extract_l3_features(graph, T0):
    features = {}
    paths = extract_length_3_paths_after_edge_removal(graph)
    content_r_ui_mean, content_r_ui_std= calculate_category_features(graph)
    last_tu, last_ti = last_t(graph, T0)
    features["content_r_ui_mean_uir"] = content_r_ui_mean
    features["content_r_ui_std_uir"] = content_r_ui_std
    features['katz_L3_uir'] = count_length_3_paths(paths)
    features['CN_L3_uir'] = calculate_CN_l3_similarity(graph)
    features['Jaccard_L3_uir'] = calculate_jaccard_l3_similarity(graph)
    features['CN_L3_dynamique_uir'] = calculate_CN_l3_dynamique(graph, T0)
    features['Jaccard_L3_dynamique_uir'] = calculate_jaccard_l3_dynamique(graph, T0)
    features['duree_L3_uir'] = calculate_duration_of_length_3_paths(graph)
    features['age_L3_uir'] = calculate_age_of_length_3_paths(graph)
    features['Katz_L3_dynamique_uir'] = calculate_katz_l3_dynamique(graph, T0)
    features['last_tu_uir'] = last_tu
    features['last_ti_uir'] = last_ti
    features['last_tu_ti_diff_uir'] = calculate_last_tu_ti_diff(graph, T0)
    return features
