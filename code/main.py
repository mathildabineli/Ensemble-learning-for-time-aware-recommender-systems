import pandas as pd
import os
import sys
import pickle
from linkstream import *
from l3_features import *
from data import *
from models import *

# Dossier contenant les jeux de données
base_dir = os.path.abspath(os.path.dirname(__file__))
folder_path = os.path.join(base_dir, '..', 'datasets')
output_path = os.path.join(base_dir, '..', 'output')

os.makedirs(output_path, exist_ok=True)

user_col = 'userId'
item_col = 'itemId'
target_col = 'rating'
k = 10

if __name__ == "__main__":
    # Liste des fichiers .csv (non enrichis)
    dataset_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and not f.startswith(('train_', 'test_', 'enriched_'))
    ])

    if len(sys.argv) != 2:
        print("Usage: python3 main.py <index>")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
        dataset_file = dataset_files[index]
    except (ValueError, IndexError):
        print(f"Indice invalide. Valeurs possibles : 0 à {len(dataset_files) - 1}")
        sys.exit(1)

    file_path = os.path.join(folder_path, dataset_file)
    dataset_name = dataset_file.replace('.csv', '')

    try:
        # Lecture du fichier
        df = pd.read_csv(file_path)
        df = add_prefixes_to_ids(df, user_col, item_col, user_prefix="U", item_prefix="I")

        # Split train/test
        train_df, test_df = train_test_split(df)

        train_path = os.path.join(output_path, f"train_{dataset_name}.csv")
        test_path = os.path.join(output_path, f"test_{dataset_name}.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Graphes
        train_graph_path = os.path.join(output_path, f"bipartite_train_{dataset_name}.pkl")
        test_graph_path = os.path.join(output_path, f"bipartite_test_{dataset_name}.pkl")
        create_bipartite_graph(train_df, train_graph_path)
        create_bipartite_graph(test_df, test_graph_path)

        with open(train_graph_path, "rb") as f:
            train_graph = pickle.load(f)
        with open(test_graph_path, "rb") as f:
            test_graph = pickle.load(f)

        train_link_times = calculate_link_times(train_graph)
        test_link_times = calculate_link_times(test_graph)

        train_users = set(train_df[user_col])
        train_items = set(train_df[item_col])
        test_users = set(test_df[user_col])
        test_items = set(test_df[item_col])

        train_link_stream = [
            (row[user_col], row[item_col], pd.to_datetime(row['timestamp']).timestamp(), row['rating'])
            for _, row in train_df.iterrows()
        ]
        test_link_stream = [
            (row[user_col], row[item_col], pd.to_datetime(row['timestamp']).timestamp(), row['rating'])
            for _, row in test_df.iterrows()
        ]

        T_train = calculate_graph_period(train_link_times)
        T_test = calculate_graph_period(test_link_times)

        tasks = []

        # LSF
        train_features_lsf_path = os.path.join(output_path, f"train_{dataset_name}_features_lsf.pkl")
        test_features_lsf_path = os.path.join(output_path, f"test_{dataset_name}_features_lsf.pkl")
        tasks.append(('lsf', train_graph, None, train_features_lsf_path, train_users, train_items, train_link_stream, train_df))
        tasks.append(('lsf', test_graph, None, test_features_lsf_path, test_users, test_items, test_link_stream, test_df))

        # L3
        T0_train = [T_train / f for f in [2, 4, 8, 16]]
        T0_test = [T_test / f for f in [2, 4, 8, 16]]

        for i, T0 in enumerate(T0_train):
            path = os.path.join(output_path, f"train_{dataset_name}_features_l3_T{i}.pkl")
            tasks.append(('l3', train_graph, T0, path, None, None, None, None))

        for i, T0 in enumerate(T0_test):
            path = os.path.join(output_path, f"test_{dataset_name}_features_l3_T{i}.pkl")
            tasks.append(('l3', test_graph, T0, path, None, None, None, None))

        print(f"Extraction des features LSF et L3 pour {dataset_name}...")

        for task in tasks:
            try:
                process_features(task)
            except Exception as e:
                print(f"Erreur pendant process_features sur {task[0]}: {e}")

        # Chargement des features
        with open(train_features_lsf_path, 'rb') as f:
            train_features_lsf = pickle.load(f)
        with open(test_features_lsf_path, 'rb') as f:
            test_features_lsf = pickle.load(f)

        train_features_l3 = {}
        for i in range(len(T0_train)):
            path = os.path.join(output_path, f"train_{dataset_name}_features_l3_T{i}.pkl")
            with open(path, 'rb') as f:
                feat = pickle.load(f)
                for feature_name, values in feat.items():
                    train_features_l3[f"{feature_name}_T{i}"] = values

        test_features_l3 = {}
        for i in range(len(T0_test)):
            path = os.path.join(output_path, f"test_{dataset_name}_features_l3_T{i}.pkl")
            with open(path, 'rb') as f:
                feat = pickle.load(f)
                for feature_name, values in feat.items():
                    test_features_l3[f"{feature_name}_T{i}"] = values

        train_all_features = {**train_features_lsf, **train_features_l3}
        test_all_features = {**test_features_lsf, **test_features_l3}

        for feature_name, feature_dict in train_all_features.items():
            train_df[feature_name] = train_df.apply(lambda row: feature_dict.get((row[user_col], row[item_col]), 0), axis=1)
        for feature_name, feature_dict in test_all_features.items():
            test_df[feature_name] = test_df.apply(lambda row: feature_dict.get((row[user_col], row[item_col]), 0), axis=1)

        enriched_train_path = os.path.join(output_path, f"enriched_train_{dataset_name}.csv")
        enriched_test_path = os.path.join(output_path, f"enriched_test_{dataset_name}.csv")
        train_df.to_csv(enriched_train_path, index=False)
        test_df.to_csv(enriched_test_path, index=False)

        print(f"{dataset_file} enrichi avec succès.")

        print(f"\n=== Entraînement du modèle sur {dataset_name} ===")
        run_models(train_df, test_df, dataset_name)

    except Exception as e:
        print(f"Erreur lors du traitement de {dataset_file} : {e}")
