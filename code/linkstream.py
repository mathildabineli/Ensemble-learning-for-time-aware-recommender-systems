import networkx as nx
import numpy as np
import pandas as pd

def extract_temporal_degrees(G, node_set):
    degrees_temporels = {}
    all_degrees = []

    for node in node_set:
        degrees_temporels[node] = {}
        timestamps = sorted(set([
            ts if not isinstance(ts, pd.Timestamp) else ts.timestamp()
            for _, _, ts in G.edges(node, data='timestamp')
        ]))

        for timestamp in timestamps:
            neighbors = list(G.neighbors(node))
            degree = len([
                neighbor for neighbor in neighbors
                if (G.edges[node, neighbor]['timestamp'].timestamp()
                    if isinstance(G.edges[node, neighbor]['timestamp'], pd.Timestamp)
                    else G.edges[node, neighbor]['timestamp']) == timestamp
            ])
            degrees_temporels[node][timestamp] = degree
            all_degrees.append(degree)

    max_degree = max(all_degrees)
    min_degree = min(all_degrees)
    if max_degree == min_degree:
        max_degree = 1.0
        min_degree = 0.0

    for node in degrees_temporels:
        for timestamp in degrees_temporels[node]:
            normalized = (degrees_temporels[node][timestamp] - min_degree) / (max_degree - min_degree)
            degrees_temporels[node][timestamp] = normalized

    return degrees_temporels


def compute_max_and_average_temporal_degrees(temporal_degrees, data):
    max_degrees = {}
    average_degrees = {}

    timestamps = data['timestamp'].apply(lambda x: x.timestamp() if isinstance(x, pd.Timestamp) else x)
    delta_time = (max(timestamps) - min(timestamps))

    for node, degrees in temporal_degrees.items():
        values = list(degrees.values())
        if len(values) > 1:
            max_degree = max(values)
            average_degree = sum(values) / delta_time if delta_time > 0 else 0
        else:
            max_degree = values[0]
            average_degree = max_degree
        max_degrees[node] = max_degree
        average_degrees[node] = average_degree

    max_vals = list(max_degrees.values())
    avg_vals = list(average_degrees.values())

    if max(max_vals) != min(max_vals):
        a, b = min(max_vals), max(max_vals)
        max_degrees = {n: (v - a) / (b - a) for n, v in max_degrees.items()}
    else:
        max_degrees = {n: 1.0 for n in max_degrees}

    if max(avg_vals) != min(avg_vals):
        a, b = min(avg_vals), max(avg_vals)
        average_degrees = {n: (v - a) / (b - a) for n, v in average_degrees.items()}
    else:
        average_degrees = {n: 1.0 for n in average_degrees}

    return max_degrees, average_degrees


def compute_assortativity(graph, users, items):
    assortativity = {}
    for u in users:
        for i in items:
            if i in graph[u]:
                deg_u = len(graph[u])
                deg_i = len(graph[i])
                assortativity[(u, i)] = min(deg_u, deg_i) / max(deg_u, deg_i)
    return assortativity


def compute_links_times(graph, node_set):
    link_times = {}
    for node in node_set:
        links = [
            (u, v, data['timestamp'].timestamp() if isinstance(data['timestamp'], pd.Timestamp) else data['timestamp'])
            for u, v, data in graph.edges(node, data=True)
        ]
        link_times[node] = sorted(links, key=lambda x: x[2])
    return link_times


def compute_inter_event_times(link_times):
    inter_event_times = {}
    for node, links in link_times.items():
        if len(links) <= 1:
            inter_event_times[node] = [0]
        else:
            times = [l[2] for l in links]
            inter_event_times[node] = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return inter_event_times


def compute_max_intercontact_time(inter_event_times):
    max_times = {n: max(t) for n, t in inter_event_times.items()}
    vals = list(max_times.values())
    a, b = min(vals), max(vals)
    if b != a:
        return {n: (v - a) / (b - a) for n, v in max_times.items()}
    else:
        return {n: 1.0 for n in max_times}


def compute_min_intercontact_time(inter_event_times):
    min_times = {n: min(t) for n, t in inter_event_times.items()}
    vals = list(min_times.values())
    a, b = min(vals), max(vals)
    if b != a:
        return {n: (v - a) / (b - a) for n, v in min_times.items()}
    else:
        return {n: 1.0 for n in min_times}


def compute_mean_intercontact_time(inter_event_times):
    mean_times = {n: np.mean(t) for n, t in inter_event_times.items()}
    vals = list(mean_times.values())
    a, b = min(vals), max(vals)
    if b != a:
        return {n: (v - a) / (b - a) for n, v in mean_times.items()}
    else:
        return {n: 1.0 for n in mean_times}


def compute_std_intercontact_time(inter_event_times):
    std_times = {n: np.std(t) for n, t in inter_event_times.items()}
    vals = list(std_times.values())
    a, b = min(vals), max(vals)
    if b != a:
        return {n: (v - a) / (b - a) for n, v in std_times.items()}
    else:
        return {n: 1.0 for n in std_times}


def transform_bipartite_graph(bipartite_graph):
    G_proj = nx.Graph()
    for v in bipartite_graph:
        neighbors = list(bipartite_graph[v])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                G_proj.add_edge(neighbors[i], neighbors[j])
    return G_proj


def argument_max(P, X, graph):
    max_inter = -1
    node = None
    for u in P | X:
        N_u = graph[u] if u in graph else set()
        inter = len(P & set(N_u))
        if inter > max_inter:
            max_inter = inter
            node = u
    return node


def bron_kerbosch_pivot(R, P, X, graph, cliques):
    if not P and not X:
        cliques.append(R)
        return
    u = argument_max(P, X, graph)
    neighbors_u = set(graph[u]) if u in graph else set()
    for v in P - neighbors_u:
        neighbors_v = set(graph[v]) if v in graph else set()
        bron_kerbosch_pivot(R + [v], P & neighbors_v, X & neighbors_v, graph, cliques)

        P.remove(v)
        X.add(v)



def maximal_clique_enum_bron_kerbosch(graph):
    cliques = []
    bron_kerbosch_pivot([], set(graph.nodes()), set(), graph, cliques)
    return cliques


def calculate_Cu(cliques, nodes):
    return {n: sum(n in c for c in cliques) for n in nodes}


def calculate_balancedness(cliques, Cu_dict, users, items):
    balancedness = {}
    for node, Cu in Cu_dict.items():
        if Cu == 0:
            balancedness[node] = 0.0
        else:
            val = 0.0
            for c in cliques:
                if node in c:
                    U = [n for n in c if n in users]
                    I = [n for n in c if n in items]
                    val += min(len(U), len(I)) / max(len(U), len(I)) if max(len(U), len(I)) > 0 else 0
            balancedness[node] = val / Cu
    return balancedness


def calculate_clique_fraction(cliques, nodes):
    total = len(cliques)
    return {n: sum(n in c for c in cliques) / total for n in nodes}


def clique_durations(cliques, link_stream):
    durations = []
    for clique in cliques:
        ts = [link[2].timestamp() if isinstance(link[2], pd.Timestamp) else link[2]
              for link in link_stream if link[0] in clique and link[1] in clique]
        if ts:
            durations.append(max(ts) - min(ts))
        else:
            durations.append(0)
    return durations


def calculate_normalized_average_duration(cliques, durations, nodes, data):
    total_duration = (max(data['timestamp']) - min(data['timestamp'])).total_seconds()
    res = {}
    for n in nodes:
        val, count = 0.0, 0
        for clique, d in zip(cliques, durations):
            if n in clique:
                val += d / total_duration if total_duration > 0 else 0
                count += 1
        res[n] = val / count if count > 0 else 0.0
    return res


def extract_lsf_features(G, users, items, link_stream, data):
    user_td = extract_temporal_degrees(G, users)
    item_td = extract_temporal_degrees(G, items)

    user_max, user_avg = compute_max_and_average_temporal_degrees(user_td, data)
    item_max, item_avg = compute_max_and_average_temporal_degrees(item_td, data)

    assort = compute_assortativity(G, users, items)

    user_lt = compute_links_times(G, users)
    item_lt = compute_links_times(G, items)

    user_iet = compute_inter_event_times(user_lt)
    item_iet = compute_inter_event_times(item_lt)

    features = {
        'user_temporal_degrees_lsf': user_td,
        'item_temporal_degrees_lsf': item_td,
        'user_max_degrees_lsf': user_max,
        'user_average_degrees_lsf': user_avg,
        'item_max_degrees_lsf': item_max,
        'item_average_degrees_lsf': item_avg,
        'assortativities_lsf': assort,
        'users_max_itc_lsf': compute_max_intercontact_time(user_iet),
        'items_max_itc_lsf': compute_max_intercontact_time(item_iet),
        'users_min_itc_lsf': compute_min_intercontact_time(user_iet),
        'items_min_itc_lsf': compute_min_intercontact_time(item_iet),
        'users_mean_itc_lsf': compute_mean_intercontact_time(user_iet),
        'items_mean_itc_lsf': compute_mean_intercontact_time(item_iet),
        'users_std_itc_lsf': compute_std_intercontact_time(user_iet),
        'items_std_itc_lsf': compute_std_intercontact_time(item_iet),
    }

    G1 = transform_bipartite_graph(G)
    cliques = maximal_clique_enum_bron_kerbosch(G1)

    Cu_users = calculate_Cu(cliques, users)
    Cu_items = calculate_Cu(cliques, items)

    durations = clique_durations(cliques, link_stream)

    features.update({
        'Cu_users_lsf': Cu_users,
        'Cu_items_lsf': Cu_items,
        'users_balancedness_lsf': calculate_balancedness(cliques, Cu_users, users, items),
        'items_balancedness_lsf': calculate_balancedness(cliques, Cu_items, users, items),
        'user_fraction_of_cliques_lsf': calculate_clique_fraction(cliques, users),
        'item_fraction_of_cliques_lsf': calculate_clique_fraction(cliques, items),
        'users_normalized_average_duration_lsf': calculate_normalized_average_duration(cliques, durations, users, data),
        'items_normalized_average_duration_lsf': calculate_normalized_average_duration(cliques, durations, items, data),
    })

    return features
