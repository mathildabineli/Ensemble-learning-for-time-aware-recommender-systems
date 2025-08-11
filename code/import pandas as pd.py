import pandas as pd


df = pd.read_csv('/home/hiperdas/Documents/experimentations_Bineli/datasets/enriched_train_handmadeprod.csv')
data = pd.read_csv('/home/hiperdas/Documents/experimentations_Bineli/datasets/enriched_test_handmadeprod.csv')
print(f"NaN in train: {df.isnull().sum().sum()}, in test: {data.isnull().sum().sum()}")


# Affiche les colonnes contenant au moins un NaN
cols_with_nan_df = df.columns[df.isnull().any()]
print("Colonnes avec NaN :", list(cols_with_nan_df))

# Affiche les colonnes contenant au moins un NaN
cols_with_nan_data = data.columns[data.isnull().any()]
print("Colonnes avec NaN :", list(cols_with_nan_data))

nan_counts_df = df.isnull().sum()
print(nan_counts_df[nan_counts_df > 0])

nan_counts_data = data.isnull().sum()
print(nan_counts_data[nan_counts_data > 0])