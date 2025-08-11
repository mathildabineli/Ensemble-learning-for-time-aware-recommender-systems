import os
import pandas as pd

# Chemin vers le dossier contenant les jeux de données
folder_path = '/home/hiperdas/Documents/experimentations_Bineli/datasets'

# Parcours de tous les fichiers du dossier
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith('.txt'):

        dataset_name = filename.replace('.txt', '')
        file_path = os.path.join(folder_path, filename)
        print(f"Traitement de {dataset_name}...")

        # Lecture du fichier
        with open(file_path, "r") as file:
            lines = file.readlines()

        data = [line.strip().split() for line in lines if len(line.strip().split()) == 5]

        if not data:
            print(f"Aucune donnée valide dans {filename}")
            continue

        columns = ['userId', 'itemId', 'categoryId', 'rating', 'timestamp']
        df = pd.DataFrame(data, columns=columns)

        # Typage selon si c'est movielens ou non
        if 'movielens' in filename.lower():
            dtype_dict = {
                'userId': int,
                'itemId': int,
                'categoryId': str,
                'rating': float,
                'timestamp': float
            }
        else:
            dtype_dict = {
                'userId': int,
                'itemId': int,
                'categoryId': int,
                'rating': float,
                'timestamp': float
            }

        df = df.astype(dtype_dict)

        # Statistiques utilisateur
        user_stats = df.groupby('userId')['rating'].agg(
            user_mean_rating='mean',
            user_median_rating='median',
            user_std_rating='std',
            user_min_rating='min',
            user_max_rating='max',
            user_num_ratings='count'
        ).reset_index()

        # Statistiques item
        item_stats = df.groupby('itemId')['rating'].agg(
            item_mean_rating='mean',
            item_median_rating='median',
            item_std_rating='std',
            item_min_rating='min',
            item_max_rating='max',
            item_num_ratings='count'
        ).reset_index()

        # Fusion des stats
        df_with_user_stats = pd.merge(df, user_stats, on='userId', how='left')
        df_with_full_stats = pd.merge(df_with_user_stats, item_stats, on='itemId', how='left')

        # One-hot encoding sur categoryId
        df_encoded = pd.get_dummies(df_with_full_stats, columns=['categoryId'], prefix='cat')
        # Identifier les colonnes one-hot
        cat_columns = [col for col in df_encoded.columns if col.startswith('cat_')]

        # Vérifier s'il y a des NaN dans ces colonnes
        print(df_encoded[cat_columns].isnull().sum())

        # Remplacer les NaN par 0, puis convertir en int
        df_encoded[cat_columns] = df_encoded[cat_columns].fillna(0).astype(int)

        # Sauvegarde CSV (fichier enrichi avec one-hot encoding)
        output_path = os.path.join(folder_path, f"{dataset_name}.csv")
        df_encoded.to_csv(output_path, index=False)
        print(f"{dataset_name} enrichi et sauvegardé sous {output_path}\n")
