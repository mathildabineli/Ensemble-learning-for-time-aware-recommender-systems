import os
import pandas as pd
import numpy as np

"""

def convert_timestamp_column(df):
    Convertit correctement la colonne timestamp.
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

# Chemin vers le dossier contenant les jeux de données
folder_path = '/home/hiperdas/Documents/experimentations_Bineli/datasets'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        dataset_name = filename.replace('.csv', '')

        try:
            df = pd.read_csv(file_path)
            train_df, test_df = train_test_split(df)

            train_output_path = os.path.join(folder_path, f"train_{dataset_name}.csv")
            test_output_path = os.path.join(folder_path, f"test_{dataset_name}.csv")

            train_df.to_csv(train_output_path, index=False)
            test_df.to_csv(test_output_path, index=False)

            print(f"{filename} traité avec succès.")

        except Exception as e:
            print(f"Erreur pour {filename} : {e}")



# list_datasets.py
import os

folder_path = os.path.join(os.path.dirname(__file__), '..', 'datasets')

dataset_files = sorted([
    f for f in os.listdir(folder_path)
    if f.endswith('.csv') and not f.startswith(('train_', 'test_', 'enriched_'))
])

for i, f in enumerate(dataset_files):
    print(f"[{i}] {f}")

"""    
import multiprocessing
print(multiprocessing.cpu_count())  # ex : 8
