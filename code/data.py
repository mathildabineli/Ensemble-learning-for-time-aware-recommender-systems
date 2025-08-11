import pandas as pd
import networkx as nx
import numpy as np
import pickle
from linkstream import *
from l3_features import *



# Dictionnaire de correspondance des noms de colonnes
"""COLONNES_STANDARD = {
    'userId': 'userId',
    'id': 'userId',
    'user': 'userId',
    'movieId': 'itemId', 
    'itemid':'itemId'
    # Ajoutez d'autres colonnes selon vos besoins
    
}

def normaliser_colonnes(df):
    # Renommer les colonnes selon le dictionnaire de correspondance
    df.rename(columns=lambda x: COLONNES_STANDARD.get(x, x), inplace=True)

    return df
"""



def add_prefixes_to_ids(df, user_col, item_col, user_prefix, item_prefix):
    """
    Ajoute des préfixes aux identifiants des utilisateurs et des items dans un DataFrame.
    """
    df_copy = df.copy()
    df_copy[item_col] = item_prefix + df_copy[item_col].astype(str)
    df_copy[user_col] = user_prefix + df_copy[user_col].astype(str)
    return df_copy


def convert_timestamp_column(df):
    """Convertit correctement la colonne timestamp."""
    if 'timestamp' not in df.columns:
        raise ValueError("Colonne 'timestamp' manquante dans le fichier.")

    # Si la colonne est de type numérique : supposer epoch
    if np.issubdtype(df['timestamp'].dtype, np.number):
        # Essayer secondes -> datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        except:
            # Si ce sont des millisecondes
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        # Si déjà string ou datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if df['timestamp'].isnull().all():
        raise ValueError("Conversion échouée : toutes les dates sont nulles.")

    return df



def train_test_split(data):
    data = convert_timestamp_column(data)
    data = data.sort_values(by='timestamp')

    min_ts = data['timestamp'].min()
    max_ts = data['timestamp'].max()
    total_days = (max_ts - min_ts).days + 1

    print(min_ts, max_ts)

    if total_days == 182:
        train_days, test_days = 140, 42
    elif total_days == 364:
        train_days, test_days = 280, 84
    elif total_days == 728:
        train_days, test_days = 560, 168
    elif total_days == 727:
        train_days, test_days = 560, 167    
    else:
        raise ValueError(f"Durée du dataset non supportée : {total_days} jours.")

    train_end_date = min_ts + pd.Timedelta(days=train_days - 1)

    train_df = data[data['timestamp'] <= train_end_date]
    test_df = data[data['timestamp'] > train_end_date]

    return train_df, test_df





def create_bipartite_graph(data, output_path) -> nx.Graph:
    link_stream = []
    for _, row in data.iterrows():
        user_id = row['userId']
        item_id = row['itemId']
        timestamp = row['timestamp']
        rating = row['rating']  
        link_stream.append((user_id, item_id, timestamp, rating))

    link_stream.sort(key=lambda x: x[2])

    G = nx.Graph()
    user_nodes = data['userId'].unique()
    G.add_nodes_from(user_nodes, bipartite=0, label='user')
    item_nodes = data['itemId'].unique()
    G.add_nodes_from(item_nodes, bipartite=1, label='item')

    for link in link_stream:
        G.add_edge(link[0], link[1], timestamp=link[2], rating=link[3])

    with open(output_path, "wb") as file:
        pickle.dump(G, file)

    return G




def process_lsf_graph(graph, users, items, link_stream, data, output_path):
    features_lsf = extract_lsf_features(graph, users, items, link_stream, data)
    with open(output_path, 'wb') as f:
        pickle.dump(features_lsf, f)


def process_l3_graph(graph, T0, penalty_func_name, output_path):
    features_l3 = extract_l3_features(graph, T0)
    with open(output_path, 'wb') as f:
        pickle.dump(features_l3, f)





def process_features(task):
    """
    task : tuple (feature_type, graph, T0, penalty_func_name, output_path)
    feature_type: 'lsf' or 'l3'
    For lsf, T0 and penalty_func_name can be None
    """
    feature_type, graph, T0, output_path, users, items, link_stream, data = task

    if feature_type == 'lsf':
        features = extract_lsf_features(graph, users, items, link_stream, data) 
    elif feature_type == 'l3':
        features = extract_l3_features(graph, T0)
    else:
        raise ValueError(f"Unknown feature_type {feature_type}")

    with open(output_path, 'wb') as f:
        pickle.dump(features, f)


"""
def process_features_for_multiple_T0(graph, T0):
    
    Calcule les features L3 pour plusieurs T0 et retourne un dictionnaire avec toutes les features combinées.
    
    T0_values = [T/2, T/4, T/8, T/16]
    all_features = {}

    for T0 in T0_values:
        features_l3 = extract_l3_features(graph, T0, penalty_func_name='exponential')
        suffix = f"T{int(T/T0)}"
        for feat_name, feat_dict in features_l3.items():
            suffixed_name = f"{feat_name}_{suffix}"
            all_features[suffixed_name] = feat_dict

    return all_features
    """