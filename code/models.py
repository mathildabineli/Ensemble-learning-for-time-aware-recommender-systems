import pandas as pd
import numpy as np
import os
import shap
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


user_col = 'userId'
item_col = 'itemId'
target_col = 'rating'
k = 10

base_dir = os.path.abspath(os.path.dirname(__file__))
output_path =  os.path.join(base_dir, '..', 'output')

# --- Catégories de features ---

def is_uir_T0(f):
    return any(key in f for key in ['_uir_T0'])

def is_uir_T1(f):
    return any(key in f for key in ['_uir_T1'])

def is_uir_T2(f):
    return any(key in f for key in ['_uir_T2'])

def is_uir_T3(f):
    return any(key in f for key in ['_uir_T3'])

def is_lsf(f):
    return f.endswith('_lsf')

def is_ord(f):
    return not (is_lsf(f) or is_uir_T0(f) or is_uir_T1(f) or is_uir_T2(f) or is_uir_T3(f))

# --- Modèles ---
MODELS = {

    "mlp": {
    "model": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,
        max_iter=10000,
        random_state=42
    ), 
    "params": {}, 
    "use_scaled": True
},


        "decision_tree": {
        "model": DecisionTreeRegressor(),
        "params": {
            'max_depth': Integer(3, 100),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        },
        "use_scaled": True
    },


    "random_forest": {
        "model": RandomForestRegressor(),
        "params": {
            'n_estimators': Integer(10, 1000),
            'max_depth': Integer(3, 100),
            'min_samples_split': Integer(2, 100),
            'min_samples_leaf': Integer(1, 100)
        },
        "use_scaled": True
    },
    "xgboost": {
        "model": XGBRegressor(),
        "params": {
            'alpha': Real(0.0, 1.0),
            'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
            'min_child_weight': Integer(0, 50),
            'max_depth': Integer(3, 100),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Integer(0, 50)
        },
        "use_scaled": True
    },


            "svm": {
        "model": SVR(),
        "params": {
            'C': Real(0.1, 100, prior='log-uniform'),
            'epsilon': Real(0.01, 1, prior='log-uniform'),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
            'gamma': Categorical(['scale', 'auto'])
        },
        "use_scaled": True
    }
}



# --- Métriques ---
def f1_at_k(y_true, y_score, k):
    top_k_idx = np.argsort(y_score)[::-1][:k]
    top_k_pred = np.zeros_like(y_true)
    top_k_pred[top_k_idx] = 1

    tp = np.sum((top_k_pred == 1) & (y_true == 1))
    fp = np.sum((top_k_pred == 1) & (y_true == 0))
    fn = np.sum((top_k_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])

    dcg = np.sum((2 ** y_true_sorted - 1) / np.log2(np.arange(2, y_true_sorted.size + 2)))
    ideal_sorted = np.sort(y_true)[::-1][:k]
    idcg = np.sum((2 ** ideal_sorted - 1) / np.log2(np.arange(2, ideal_sorted.size + 2)))
    return dcg / (idcg + 1e-8)


def prepare_features(df, feature_groups):
    excluded_cols = {user_col, item_col}
    X = {}
    for group_name, func in feature_groups.items():
        feats = [f for f in df.columns if func(f) and f not in excluded_cols]
        X[group_name] = df[feats].values
    return X

# --- Entraînement et évaluation ---
def run_models(train_df, test_df, dataset_name):

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp']).astype(np.int64) / 1e9
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp']).astype(np.int64) / 1e9


    # Remplacer les NaN par 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    feature_groups = {
        "ord": is_ord,
        "lsf": is_lsf,
        "uir_t2": is_uir_T0,
        "ord+lsf": lambda f: is_ord(f) or is_lsf(f),
        "ord+uir_t2": lambda f: is_ord(f) or is_uir_T0(f),
        "lsf+uir_t2": lambda f: is_lsf(f) or is_uir_T0(f),
        "ord+lsf+uir_t2": lambda f: True,
        "uir_t4": is_uir_T1,
        "ord+uir_t4": lambda f: is_ord(f) or is_uir_T1(f),
        "lsf+uir_t4": lambda f: is_lsf(f) or is_uir_T1(f),
        "ord+lsf+uir_t4": lambda f: True,
        "uir_t8": is_uir_T2,
        "ord+uir_t8": lambda f: is_ord(f) or is_uir_T2(f),
        "lsf+uir_t8": lambda f: is_lsf(f) or is_uir_T2(f),
        "ord+lsf+uir_t8": lambda f: True,
        "uir_t16": is_uir_T3,
        "ord+uir_t16": lambda f: is_ord(f) or is_uir_T3(f),
        "lsf+uir_t16": lambda f: is_lsf(f) or is_uir_T3(f),
        "ord+lsf+uir_t16": lambda f: True

    }

    X_train_groups = prepare_features(train_df, feature_groups)
    X_test_groups = prepare_features(test_df, feature_groups)

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    results = []
    shap_values_dict = {}

    for model_name, model_info in MODELS.items():
        for group_name in feature_groups:
            print(f"Optimizing and training {model_name} on {group_name} features...")
            X_train = X_train_groups[group_name]
            X_test = X_test_groups[group_name]

            base_model = model_info["model"]
            param_space = model_info["params"]
            use_scaled = model_info["use_scaled"]



            if model_name == "mlp":
                 best_model = Pipeline([
                     ('scaler', StandardScaler()),
                     ('model', MLPRegressor(
                         hidden_layer_sizes=(64, 32),
                         activation='relu',
                         alpha=0.001,
                         max_iter=1000,
                         random_state=42
                         ))
                 ])
                 best_model.fit(X_train, y_train)
                

            else:     
                if use_scaled:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', base_model)
                          ])
                    search = BayesSearchCV(
                        estimator=pipeline,
                        search_spaces={'model__' + k: v for k, v in param_space.items()},
                        cv=5,
                        n_iter=25,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                        )
                    
                else:
                    search = BayesSearchCV(
                        estimator=base_model,
                        search_spaces=param_space,
                        cv=5,
                        n_iter=25,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                        )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # F1@10 et NDCG@10 par utilisateur
            test_df_copy = test_df.copy()
            test_df_copy['pred'] = y_pred

            user_scores = defaultdict(list)
            user_truth = defaultdict(list)

            for row in test_df_copy.itertuples():
                user = getattr(row, user_col)
                item = getattr(row, item_col)
                pred = getattr(row, 'pred')
                rating = getattr(row, target_col)

                user_scores[user].append((item, pred))
                user_truth[user].append((item, rating))

            f1_list = []
            ndcg_list = []

            for user in user_scores:
                scores = sorted(user_scores[user], key=lambda x: x[1], reverse=True)[:k]
                true_ratings = {item: rating for item, rating in user_truth[user]}
                y_score = [pred for item, pred in scores]
                y_true = [1 if true_ratings.get(item, 0) >= 4 else 0 for item, _ in scores]

                f1_list.append(f1_at_k(np.array(y_true), np.array(y_score), k))
                ndcg_list.append(ndcg_at_k(np.array(y_true), np.array(y_score), k))

            f1 = np.mean(f1_list)
            ndcg = np.mean(ndcg_list)

            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'feature_set': group_name,
                'RMSE': rmse,
                'MAE': mae,
                f'F1@{k}': f1,
                f'NDCG@{k}': ndcg
            })

            # SHAP
            if group_name in ["ord+lsf+uir_t2", "ord+lsf+uir_t4", "ord+lsf+uir_t8", "ord+lsf+uir_t16"]:
                try:
                    if model_name == "mlp" or isinstance(best_model, Pipeline):
                        explainer = shap.KernelExplainer(best_model.predict, X_train[:100])  # approximation + rapide
                        shap_values = explainer.shap_values(X_test[:100])  # on limite à 100 échantillons pour accélérer
                        
                    else:
                        explainer = shap.Explainer(best_model, X_train[:100])
                        shap_values = explainer(X_test[:100])
                     # Sauvegarde dans la liste
                    #shap_values_dict[(dataset_name, model_name, group_name)] = shap_values
                    # Associer les noms de colonnes pour rendre le CSV lisible
                    feature_names = [f for f in train_df.columns if feature_groups[group_name](f) and f not in {user_col, item_col}]
                    shap_df = pd.DataFrame(shap_values, columns=feature_names)
                    shap_df['dataset'] = dataset_name
                    shap_df['model'] = model_name
                    shap_df['feature_set'] = group_name
                    shap_values_dict[(dataset_name, model_name, group_name)] = shap_df
                    
                except Exception as e:
                    print(f"SHAP computation failed for {model_name} on {group_name}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, f"results_{dataset_name}.csv"), index=False)

    if shap_values_dict:
        shap_concat = pd.concat(shap_values_dict.values(), ignore_index=True)
        shap_concat.to_csv(os.path.join(output_path, f"shap_values_{dataset_name}.csv"), index=False)





