# Document de compétences

**Projet** Impact du CO2 sur la température à Paris

**Titre visé** Développeur en intelligence artificielle, RNCP37827 (2023)

**Formateur** Thomas Viviani

Ce document reprend chacune des 21 compétences du titre et donne pour chacune un exemple concret tiré du projet, avec le fichier où le vérifier, lorsqu'une compétence n'est pas ou peu travaillée dans ce projet, cela est indiqué honnêtement

## Vue d'ensemble

<table>
<tr><th>Bloc</th><th>Compétence</th><th>Niveau dans le projet</th><th>Preuve principale</th></tr>
<tr><td rowspan="5">Bloc 1 Données</td><td>C1 Automatiser l'extraction</td><td>Couverte</td><td>src/etl.py, run_pipeline.py</td></tr>
<tr><td>C2 Requêtes SQL</td><td>Couverte</td><td>src/requetes.py</td></tr>
<tr><td>C3 Agrégation et nettoyage</td><td>Couverte</td><td>src/etl.py</td></tr>
<tr><td>C4 Base de données</td><td>Couverte</td><td>option1.loo, src/load_db.py</td></tr>
<tr><td>C5 API REST des données</td><td>Non couverte</td><td>remplacée par l'accès SQL direct</td></tr>
<tr><td rowspan="8">Bloc 2 Modèles IA</td><td>C6 Veille</td><td>Partielle</td><td>comparaison avec Météo France</td></tr>
<tr><td>C7 Benchmark de services IA</td><td>Partielle</td><td>choix entre ML et API LLM</td></tr>
<tr><td>C8 Paramétrage d'un service IA</td><td>Couverte</td><td>src/model.py, src/experiences_mlflow.py</td></tr>
<tr><td>C9 API REST du modèle</td><td>Non couverte</td><td>modèle exposé via Streamlit</td></tr>
<tr><td>C10 Intégration du modèle</td><td>Couverte</td><td>dashboard/dashboard.py</td></tr>
<tr><td>C11 Monitorage du modèle</td><td>Couverte</td><td>MLflow, table predictions</td></tr>
<tr><td>C12 Tests du modèle</td><td>Couverte</td><td>tests/test_projet.py</td></tr>
<tr><td>C13 Livraison continue du modèle</td><td>Partielle</td><td>run_pipeline.py</td></tr>
<tr><td rowspan="8">Bloc 3 Application</td><td>C14 Analyse du besoin</td><td>Couverte</td><td>problématique et hypothèses</td></tr>
<tr><td>C15 Conception technique</td><td>Couverte</td><td>architecture du projet</td></tr>
<tr><td>C16 Coordination agile et MLOps</td><td>Partielle</td><td>branches Git dev et master</td></tr>
<tr><td>C17 Développement des composants</td><td>Couverte</td><td>dashboard/dashboard.py</td></tr>
<tr><td>C18 Tests automatisés</td><td>Partielle</td><td>pytest dans le pipeline</td></tr>
<tr><td>C19 Livraison continue</td><td>Couverte</td><td>Docker, Streamlit Cloud</td></tr>
<tr><td>C20 Surveillance de l'application</td><td>Partielle</td><td>journaux du pipeline</td></tr>
<tr><td>C21 Résolution d'incidents</td><td>Couverte</td><td>5 incidents documentés plus bas</td></tr>
</table>

## Bloc 1 Collecte, stockage et mise à disposition des données

### C1 Automatiser l'extraction de données

**Réalisation** les données viennent de deux sources différentes, un fichier Kaggle sur le CO2 mondial et les relevés quotidiens Météo France du département 75 téléchargés sur data.gouv.fr, le script etl.py les lit avec Pandas et le script run_pipeline.py enchaîne automatiquement toutes les étapes, une seule commande reconstruit donc tout le projet à partir des fichiers bruts

**Où le voir** src/etl.py et run_pipeline.py

```python
etapes = [
    ("ETL (nettoyage des données)", "etl.py"),
    ("chargement dans DuckDB", "load_db.py"),
    ("Requêtes SQL", "requetes.py"),
    ("Analyse + modèle ml", "model.py"),
    ("Expériences MLflow", "experiences_mlflow.py"),
]
```

### C2 Développer des requêtes SQL

**Réalisation** sept requêtes interrogent le schéma en étoile avec des jointures entre la table de faits et les dimensions, des regroupements, une sous requête WITH pour calculer la normale climatique 1961 à 1990 et une fonction de fenêtre pour la moyenne mobile sur dix ans, une requête supplémentaire analyse les erreurs du modèle dans la table predictions

**Où le voir** src/requetes.py, onglet Requêtes SQL de l'application, résultats dans results/sql_*.csv

```sql
SELECT a.decennie,
       ROUND(AVG(f.avg_temp_france), 2) AS temp_moyenne_c,
       ROUND(AVG(f.co2_ppm), 1) AS co2_moyen_ppm
FROM fait_climat f
JOIN dim_annee a ON f.annee = a.annee
GROUP BY a.decennie
ORDER BY a.decennie
```

**Résultat** la température moyenne passe de 11,17 °C dans les années 1960 à 13,36 °C dans les années 2020

### C3 Développer des règles d'agrégation et de nettoyage

**Réalisation** dans les relevés Météo France, certains jours ont une température moyenne marquée comme douteuse alors que les minimales et maximales sont validées, ces jours sont recalculés comme la moyenne de la minimale et de la maximale, les relevés quotidiens sont ensuite agrégés par année, filtrés sur la période commune 1958 à 2024, puis fusionnés avec les données CO2, deux variables sont créées, la croissance annuelle du CO2 et une moyenne mobile sur cinq ans

**Où le voir** src/etl.py

```python
douteux = (meteo["QTM"] == 9) & (meteo["QTN"] == 1) & (meteo["QTX"] == 1)
meteo.loc[douteux, "TM"] = (meteo.loc[douteux, "TN"] + meteo.loc[douteux, "TX"]) / 2
```

### C4 Créer une base de données

**Réalisation** le modèle conceptuel et le modèle logique ont été réalisés avec la méthode Merise dans Looping, puis traduits en schéma en étoile dans DuckDB avec une table de faits fait_climat et deux dimensions, dim_annee et dim_classe_thermique, les clés primaires et étrangères sont déclarées, l'import est entièrement scripté, la question du RGPD ne se pose pas car les données sont publiques et ne contiennent aucune donnée personnelle

**Où le voir** option1.loo pour le MCD et le MLD, src/load_db.py pour la création physique

```sql
CREATE TABLE fait_climat (
    annee INTEGER PRIMARY KEY REFERENCES dim_annee(annee),
    id_classe INTEGER REFERENCES dim_classe_thermique(id_classe),
    co2_ppm DOUBLE,
    temp_anomaly_global DOUBLE,
    co2_growth DOUBLE,
    avg_temp_france DOUBLE
)
```

### C5 Développer une API REST pour les données

**Non couverte** dans ce projet, l'application Streamlit lit directement la base DuckDB en lecture seule, une évolution possible serait une petite API FastAPI qui exposerait les tables fait_climat et predictions

## Bloc 2 Intégration de modèles et services d'intelligence artificielle

### C6 Réaliser une veille technique et réglementaire

**Partielle** la projection obtenue pour 2035 a été comparée aux estimations officielles de Météo France, ce qui a permis d'identifier et de documenter la principale limite du modèle, la veille technique a aussi porté sur MLflow, dont la version 3 refuse par défaut d'enregistrer certains modèles, ce qui a conduit à choisir explicitement le format d'enregistrement

### C7 Identifier des services d'intelligence artificielle

**Partielle** le sujet laissait le choix entre un modèle de machine learning et une API de modèle de langage, le machine learning a été retenu car la question porte sur la prédiction d'une valeur numérique à partir de données chiffrées, un LLM n'aurait rien apporté de mesurable ici, puis trois familles de modèles ont été comparées

### C8 Paramétrer un service d'intelligence artificielle

**Réalisation** trois modèles sont entraînés et comparés sur les mêmes données, régression linéaire, Random Forest et réseau de neurones MLP, puis cinq configurations différentes sont testées dans MLflow en faisant varier le nombre d'arbres, la profondeur et la régularisation Ridge

**Où le voir** src/model.py et src/experiences_mlflow.py

<table>
<tr><th>Critère</th><th>Régression linéaire</th><th>Random Forest</th><th>Réseau de neurones</th></tr>
<tr><td>MAE</td><td>0,558 °C</td><td>1,327 °C</td><td>1,583 °C</td></tr>
<tr><td>Accuracy à 0,5 °C</td><td>36 %</td><td>21 %</td><td>14 %</td></tr>
<tr><td>Temps d'entraînement</td><td>0,003 s</td><td>0,11 s</td><td>0,34 s</td></tr>
<tr><td>Complexité du code</td><td>Faible</td><td>Faible</td><td>Moyenne</td></tr>
<tr><td>Facilité de réglage</td><td>Très facile</td><td>Facile</td><td>Difficile</td></tr>
</table>

### C9 Développer une API REST pour le modèle

**Non couverte** le modèle est chargé directement par l'application Streamlit depuis db/temp_predictor.pkl, une évolution possible serait une route FastAPI qui recevrait une valeur de CO2 et renverrait la température prédite

### C10 Intégrer le modèle dans une application

**Réalisation** l'onglet Intelligence artificielle de l'application charge le meilleur modèle sauvegardé, affiche la comparaison des modèles, la courbe réel contre prédit, l'analyse des erreurs avec un seuil réglable, et permet de faire une prédiction sur une nouvelle valeur de CO2 ou pour une année choisie entre 2025 et 2060

**Où le voir** dashboard/dashboard.py, fonction charger_modele et onglet Intelligence artificielle

### C11 Monitorer un modèle

**Réalisation** chaque expérience est tracée dans MLflow avec ses paramètres, quatre métriques (accuracy, MAE, RMSE, R²), son temps d'entraînement et le modèle lui même, toutes les prédictions du jeu de test sont aussi historisées dans la table SQL predictions, ce qui permet de compter les erreurs par modèle

**Où le voir** src/experiences_mlflow.py, la base mlflow.db consultable avec mlflow ui, la table predictions

```sql
SELECT modele, COUNT(*) AS nb_predictions,
       COUNT(*) FILTER (WHERE ABS(prediction - reel) > 0.5) AS nb_erreurs
FROM predictions
GROUP BY modele
```

**Ce que l'on a observé** augmenter le nombre d'arbres ou leur profondeur n'améliore pas la Random Forest, la régression linéaire reste la meilleure avec 9 erreurs sur 14 contre 11 et 12 pour les deux autres modèles

### C12 Programmer les tests automatisés du modèle

**Réalisation** quinze tests pytest vérifient les données nettoyées (67 années, aucune valeur manquante), la structure du schéma en étoile, le respect des clés étrangères, la plausibilité des valeurs, le bon fonctionnement de chaque requête SQL, la cohérence de la table predictions, le calcul des métriques, et que le modèle sauvegardé prédit une température réaliste qui augmente avec le CO2

**Où le voir** tests/test_projet.py, lancement avec python et le module pytest

### C13 Créer une chaîne de livraison continue du modèle

**Partielle** le script run_pipeline.py automatise la chaîne complète, données, base, entraînement, expériences, tests, et s'arrête dès qu'une étape ou un test échoue, ce qui empêche de livrer un modèle cassé, il n'est pas encore branché sur un outil d'intégration continue comme GitHub Actions

## Bloc 3 Application intégrant un service d'intelligence artificielle

### C14 Analyser le besoin

**Réalisation** le besoin a été formulé sous forme d'une problématique claire et de trois hypothèses vérifiables, chaque hypothèse reçoit une réponse argumentée dans le README, dont une hypothèse infirmée

**Où le voir** README.md, onglet Présentation de l'application

### C15 Concevoir le cadre technique

**Réalisation** l'architecture sépare clairement les données brutes et nettoyées, les scripts de traitement, la base, les résultats, les tests et l'application, les outils ont été choisis pour leur simplicité, DuckDB qui ne demande aucun serveur, scikit learn, MLflow, Streamlit et Plotly

**Où le voir** section Architecture du README

### C16 Coordonner la réalisation technique

**Partielle** le projet est versionné avec Git et hébergé sur GitHub, avec une branche dev pour le développement et une branche master pour la version livrée, la démarche MLOps se retrouve dans la traçabilité MLflow et le pipeline reproductible

### C17 Développer les composants de l'application

**Réalisation** l'application Streamlit comporte cinq onglets, Présentation, Exploration avec trois filtres (plage d'années, périodes, classe thermique) et quatre graphiques interactifs accompagnés de leur interprétation, Requêtes SQL, Intelligence artificielle et MLOps, la base est ouverte en lecture seule et les lectures sont mises en cache pour la rapidité

**Où le voir** dashboard/dashboard.py

### C18 Automatiser les tests du code

**Partielle** les tests pytest sont lancés automatiquement par le pipeline à chaque exécution, l'étape suivante serait de les faire tourner à chaque envoi sur GitHub grâce à un outil d'intégration continue

### C19 Mettre en place la livraison continue

**Réalisation** l'application est conteneurisée avec un Dockerfile et un fichier docker compose, et elle est déployée sur Streamlit Community Cloud, qui republie automatiquement l'application à chaque envoi sur la branche master de GitHub

**Où le voir** le Dockerfile, le fichier de configuration docker compose et l’[application en ligne](https://projet-consolidation.streamlit.app/)

### C20 Surveiller l'application

**Partielle** le pipeline affiche un journal de chaque étape et signale l'étape qui échoue, l'application gère le cas où les filtres ne renvoient aucune donnée en affichant un avertissement plutôt qu'une erreur, il n'y a pas encore d'outil de journalisation dédié

### C21 Résoudre des incidents techniques

Cinq incidents réels ont été rencontrés et corrigés pendant le projet

<table>
<tr><th>Incident</th><th>Cause</th><th>Correction</th></tr>
<tr><td>Le pipeline plantait dès la première étape</td><td>etl.py enregistrait dans un dossier data brut qui n'existe pas, alors que load_db.py lisait data/clean</td><td>chemin de sauvegarde remis sur data/clean</td></tr>
<tr><td>L'image Docker contenait environ 100 Mo inutiles</td><td>le fichier s'appelait dockerignore sans le point initial, Docker l'ignorait</td><td>renommé en .dockerignore et complété avec les données brutes et le dossier mlruns</td></tr>
<tr><td>La Random Forest prédisait la même valeur pour toutes les années récentes</td><td>un arbre de décision ne sait pas extrapoler au delà des valeurs vues à l'entraînement</td><td>découpage chronologique du jeu de test pour rendre le problème visible, choix de la régression linéaire comme modèle final</td></tr>
<tr><td>MLflow refusait d'enregistrer la Random Forest</td><td>MLflow 3 utilise par défaut un format sécurisé qui n'accepte pas les arbres</td><td>choix explicite du format cloudpickle pour nos propres modèles</td></tr>
<tr><td>Le découpage aléatoire faisait voir le futur au modèle</td><td>train_test_split mélangeait les années</td><td>entraînement sur 1958 à 2010 et test sur 2011 à 2024</td></tr>
</table>
