# Impact du CO2 sur la température à Paris

Projet de data science complet qui enchaîne nettoyage des données, base DuckDB en étoile, requêtes SQL, comparaison de modèles de machine learning, suivi MLflow, tests automatiques et application Streamlit

[Voir l'application en ligne](https://projet-consolidation.streamlit.app/)

## Problématique

Nous cherchons à savoir si le CO2 influence les températures à Paris, et s'il permet de prédire la température moyenne annuelle à l'horizon 2035

Cette question est une version locale et vérifiable d'un débat mondial, plutôt que de répéter que le CO2 réchauffe la planète, on mesure le lien sur des données réelles de 1958 à 2024 et on en quantifie la force

## Hypothèses testées

1) Il existe une corrélation positive entre la concentration de CO2 et la température moyenne annuelle à Paris

2) Un modèle non linéaire (Random Forest, réseau de neurones) capture mieux cette relation qu'une régression linéaire simple

3) Le réchauffement observé à Paris suit la même tendance que l'anomalie de température mondiale

## Résultats obtenus

Les modèles sont entraînés sur les années 1958 à 2010 puis testés sur les années 2011 à 2024 qu'ils n'ont jamais vues, comme pour une vraie prévision

<table>
<tr><th>Indicateur</th><th>Régression linéaire</th><th>Random Forest</th><th>Réseau de neurones MLP</th></tr>
<tr><td>MAE (erreur moyenne)</td><td>0,558 °C</td><td>1,327 °C</td><td>1,583 °C</td></tr>
<tr><td>RMSE</td><td>0,639 °C</td><td>1,500 °C</td><td>1,766 °C</td></tr>
<tr><td>R²</td><td>0,18</td><td>négatif</td><td>négatif</td></tr>
<tr><td>Accuracy à 0,5 °C près</td><td>36 %</td><td>21 %</td><td>14 %</td></tr>
<tr><td>Temps d'entraînement</td><td>0,003 s</td><td>0,11 s</td><td>0,34 s</td></tr>
<tr><td>Complexité du code</td><td>Faible</td><td>Faible</td><td>Moyenne</td></tr>
<tr><td>Facilité de réglage</td><td>Très facile</td><td>Facile</td><td>Difficile</td></tr>
</table>

<table>
<tr><th>Indicateur global</th><th>Valeur</th></tr>
<tr><td>Corrélation CO2 et température à Paris</td><td>0,64</td></tr>
<tr><td>Corrélation CO2 et anomalie mondiale</td><td>0,96</td></tr>
<tr><td>CO2 projeté en 2035</td><td>431,9 ppm</td></tr>
<tr><td>Température projetée en 2035</td><td>13,39 °C</td></tr>
</table>

### Réponse aux hypothèses

**Hypothèse 1 confirmée** avec une corrélation de 0,64, le lien est réel mais partiel car la météo de chaque année ajoute beaucoup de variations

**Hypothèse 2 infirmée** car la régression linéaire est nettement meilleure, la Random Forest prédit la même valeur pour toutes les années récentes puisqu'un arbre ne sait pas sortir des valeurs vues à l'entraînement, et le réseau de neurones prolonge trop fort la tendance

**Hypothèse 3 globalement confirmée** car la température de Paris et l'anomalie mondiale montent ensemble, avec une amplitude différente

**Limite assumée** notre projection pour 2035 est plus basse que les estimations officielles de Météo France, car le modèle prolonge le CO2 en ligne droite alors que les émissions réelles accélèrent, c'est une simplification volontaire

## Sources de données

<table>
<tr><th>Source</th><th>Fichier</th><th>Contenu</th></tr>
<tr><td>Kaggle, Global CO2 vs Temp Anomaly</td><td>climate_merged.csv</td><td>CO2 en ppm et anomalie de température mondiale de 1958 à 2024</td></tr>
<tr><td>Météo France sur data.gouv.fr</td><td>fichier Q_75 previous couvrant 1950 à 2024</td><td>Températures quotidiennes de toutes les stations de Paris</td></tr>
</table>

**Note méthodologique** la température annuelle est la moyenne de toutes les stations actives du département 75, et les jours dont la température moyenne est douteuse sont recalculés à partir des minimales et maximales validées, seule la station de Montsouris couvre toute la période, ce choix ajoute donc une variabilité liée au nombre de stations actives chaque année

## Architecture du projet

```
consolidation/
├── data/
│   ├── raw/                     fichiers sources bruts
│   └── clean/                   données nettoyées et matrice de corrélation
├── src/
│   ├── etl.py                   nettoyage et fusion des sources
│   ├── load_db.py               création du schéma en étoile dans DuckDB
│   ├── requetes.py              les 7 requêtes SQL d'analyse
│   ├── model.py                 comparaison de 3 modèles et table predictions
│   └── experiences_mlflow.py    5 expériences suivies avec MLflow
├── tests/
│   └── test_projet.py           15 tests automatiques
├── db/
│   ├── climat.duckdb            base de données
│   └── temp_predictor.pkl       meilleur modèle sauvegardé
├── results/                     résultats exportés en CSV
├── dashboard/
│   └── dashboard.py             application Streamlit en 5 onglets
├── docs/
│   └── document_competences.md  livrable avec un exemple par compétence
├── option1.loo                  MCD et MLD réalisés avec Looping
├── run_pipeline.py              exécute tout le projet en une commande
├── Dockerfile et docker-compose.yml
└── requirements.txt
```

## Modélisation de la base

**Merise** le MCD et le MLD sont dans option1.loo, avec les entités CO2_MONDIAL et TEMPERATURE_PARIS reliées par l'association CONCERNE

**Schéma en étoile** une table de faits fait_climat (CO2, anomalie mondiale, croissance du CO2, température de Paris) entourée de deux dimensions, dim_annee (année, décennie, période) et dim_classe_thermique (année fraîche, normale ou chaude), plus la table predictions qui stocke les sorties des modèles

## Requêtes SQL

Les requêtes sont dans src/requetes.py et consultables dans l'onglet Requêtes SQL de l'application

<table>
<tr><th>Requête</th><th>Ce qu'elle montre</th></tr>
<tr><td>nb_annees_par_classe</td><td>21 années fraîches, 25 normales et 21 chaudes</td></tr>
<tr><td>moyennes_par_decennie</td><td>la température passe de 11,2 °C dans les années 1960 à 13,4 °C dans les années 2020</td></tr>
<tr><td>evolution_par_periode</td><td>le CO2 a pris 55 ppm entre 2000 et 2024 contre 22 ppm entre 1958 et 1979</td></tr>
<tr><td>top5_annees_chaudes</td><td>4 des 5 années les plus chaudes sont postérieures à 2017</td></tr>
<tr><td>ecart_a_la_normale_1961_1990</td><td>écart de chaque année à la normale climatique de référence</td></tr>
<tr><td>croissance_co2_par_decennie</td><td>la hausse annuelle du CO2 a été multipliée par 3,5 depuis les années 1950</td></tr>
<tr><td>moyenne_mobile_10_ans</td><td>tendance lissée qui gomme les variations d'une année sur l'autre</td></tr>
</table>

L'analyse des erreurs du modèle se fait aussi en SQL dans la table predictions, une prédiction est comptée comme erreur si elle s'écarte de plus de 0,5 °C de la valeur réelle (9 erreurs sur 14 pour la régression linéaire, 11 pour la Random Forest, 12 pour le réseau de neurones)

## Expériences MLflow

<table>
<tr><th>Expérience</th><th>MAE</th><th>R²</th><th>Accuracy à 0,5 °C</th></tr>
<tr><td>Régression linéaire</td><td>0,558</td><td>0,18</td><td>36 %</td></tr>
<tr><td>Ridge avec alpha 10 000</td><td>0,619</td><td>0,04</td><td>43 %</td></tr>
<tr><td>Random Forest 50 arbres profondeur 3</td><td>1,283</td><td>négatif</td><td>21 %</td></tr>
<tr><td>Random Forest 200 arbres sans limite</td><td>1,461</td><td>négatif</td><td>14 %</td></tr>
<tr><td>Random Forest 100 arbres profondeur 5</td><td>1,484</td><td>négatif</td><td>14 %</td></tr>
</table>

**Ce que l'on observe** ajouter des arbres ou de la profondeur n'aide pas la Random Forest, plus elle est profonde plus elle colle aux années d'entraînement et moins elle généralise, la régression linéaire reste la meilleure, et une régularisation Ridge trop forte aplatit la droite, ce qui dégrade l'erreur moyenne même si un peu plus de prédictions tombent dans la marge de 0,5 °C

Chaque expérience enregistre ses paramètres, quatre métriques (accuracy, MAE, RMSE, R²), son temps d'entraînement et le modèle obtenu

## Comment exécuter le projet

Installer les bibliothèques

```bash
pip install -r requirements.txt
```

Tout lancer dans l'ordre (nettoyage, base, requêtes, modèles, MLflow, tests puis application)

```bash
python run_pipeline.py
```

Lancer seulement l'application

```bash
streamlit run dashboard/dashboard.py
```

Ouvrir l'interface MLflow depuis le dossier du projet, puis ouvrir dans le navigateur l'adresse affichée par la commande

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Lancer les tests

```bash
python -m pytest
```

Lancer avec Docker

```bash
docker compose up --build
```

## Badges visés

Data Engineer avec l'ETL, DuckDB et le pipeline reproductible, Data Architect avec Merise et le schéma en étoile, IA avec la comparaison de modèles, MLflow et les tests
