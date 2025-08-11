# Recommandation de liens bipartis avec des descripteurs temporels et structurels

Ce projet vise à évaluer des modèles de prédiction de la note (`rating`) dans des systèmes de recommandation dynamiques, en exploitant diverses catégories de descripteurs ORD, LSF, UIR, avec plusieurs modèles supervisés xgboost, random forest, decision tree, svm et mlp.

## 📁 Structure du projet
experimentations_Bineli/
├── code/ # Contient tout le code source
│ ├── data.py # Prétraitement des données
│ ├── linkstream.py # Extraction des descripteurs lsf
│ ├── l3_features.py # Extraction des descripteurs uir
│ ├── models.py # Entraînement, prédiction 
│ └── main.py # Script principal
├── datasets/ # Contient les datasets 
├── output/ # Résultats : métriques 
├── requirements.txt # Dépendances du projet
└── README.md # Ce fichier



## ⚙️ Installation

Copier le projet, puis installer les dépendances :

pip install -r requirements.txt


## ⚙️ Exécution


Depuis le dossier code/, lancer :

python main.py


Cela va :

Entraîner les modèles sur différents groupes de features

Sauvegarder les résultats dans le dossier output/ :

results_<dataset>.csv : contient les RMSE, MAE, F1@10 et NDCG@10

NB: les jeux de données sont déja prétraités


