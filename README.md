Impact du CO₂ sur la température en rance

Projet de data science de bout en bout ETL, base de données (DuckDB), modélisation Merise/Star Schema, analyse et modèle prédicti, dashboard interacti (Streamlit).

Problématique

 Comment le CO₂ inluence-t-il les températures en Île-de-rance, et peut-on prédire la température moyenne annuelle à horizon 2035 à partir des émissions mondiales de CO₂ ?

Pourquoi cette question est `intéressante`

    C'est une version locale et vériiable d'un débat mondial  plutôt que de répéter que "le CO2 réchaue la planète", on le mesure concrètement sur des données réelles (1958-2024) et on quantiie la orce du lien.

 Hypothèses testées

1. Il existe une corrélation positive et signiicative entre la concentration de CO₂ atmosphérique et la température moyenne annuelle mesurée en rance.
2. Un modèle Random orest capture mieux cette relation qu'une régression linéaire simple.
3. Le réchauement observé suit la même tendance générale que l'anomalie de température mondiale.

  Résultats obtenus

| Indicateur | Valeur |
|---|---|
| Corrélation CO2 / température | 0.64 |
| Régression linéaire — MAE | 0.555 °C |
| Régression linéaire — R² | 0.579 |
| Random orest — MAE | 0.593 °C |
| Random orest — R² | 0.542 |
| CO2 projeté en 2035 | 431.9 ppm |
| Température projetée en 2035 | 13.21 °C |

 Réponse aux hypothèses 
-  Hypothèse 1 conirmée  corrélation positive de 0.64 entre CO2 et température — un lien réel, bien que partiel (d'autres acteurs inluencent la température locale).
-  Hypothèse 2 inirmée  la régression linéaire ait mieux que le Random orest (moins d'erreur, plus de variance expliquée). Avec peu de données et une seule variable explicative, la lexibilité du Random orest n'apporte rien et généralise moins bien.
-  Hypothèse 3 globalement conirmée  la tendance de réchauement local suit la tendance mondiale, avec une magnitude diérente.

 Limite assumée  notre projection 2035 (13.21°C) est un peu plus basse que les estimations oicielles de Météo-rance (TRACC, ~14-14.5°C), car notre modèle suppose une progression linéaire du CO2 alors que les modèles climatiques oiciels intègrent une accélération des émissions. C'est une simpliication volontaire, pas une erreur cachée.

  Sources de données

| Source | ichier | Contenu |
|---|---|---|
| Kaggle — Global CO2 vs Temp Anomaly | `climate_merged.csv` | CO2 (ppm) et anomalie de température mondiale, 1958-2024 |
| Météo-rance (data.gouv.r) | `Q_75_avant-1949_RR-T-Vent.csv`, `Q_75_previous-1950-2024_RR-T-Vent.csv`, `Q_75_latest-2025-2026_RR-T-Vent.csv` | Températures quotidiennes, département 75 (Paris), toutes stations |

 Note méthodologique  la température locale est calculée en moyennant l'ensemble des stations actives du département 75 chaque année. Un contrôle qualité a été eectué (voir ETL)  seule la station Paris-Montsouris a une couverture complète sur 1958-2024 ; les 37 autres stations n'ont qu'une couverture partielle ou nulle sur cette période. Ce choix méthodologique (moyenner toutes les stations plutôt que la seule station de réérence) introduit une variabilité liée au nombre de stations actives par année, documentée ici pour transparence.

  Architecture du projet

```
consolidation/
├── data/
│   ├── raw/        # ichiers sources bruts
│   └── clean/      # données nettoyées (sortie de l'ETL)
├── src/
│   ├── etl.py      # nettoyage + usion des sources
│   ├── load_db.py  # chargement dans DuckDB (star schema)
│   └── model.py    # analyses, corrélations, modèles ML
├── db/
│   ├── climat.duckdb       # base de données
│   └── temp_predictor.pkl  # modèle entraîné sauvegardé
├── dashboard/
│   └── dashboard.py  # dashboard Streamlit
├── run_pipeline.py   # exécute tout le pipeline en une commande
├── requirements.txt
└── README.md
```

  Modélisation de la base

-  Merise (MCD/MLD)  entités `CO2_MONDIAL` et `TEMPERATURE_PARIS`, reliées par une association `CONCERNE` en cardinalité (1,1)-(1,1).
-  Star Schema   table de aits `ait_climat` (co2_ppm, temp_anomaly_global, co2_growth, avg_temp_rance) + dimension `dim_annee` (annee, decennie).

   Comment exécuter le projet

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Ce script exécute automatiquement, dans l'ordre  le nettoyage des données (`etl.py`), le chargement dans DuckDB (`load_db.py`), les analyses et l'entraînement des modèles (`model.py`), puis lance le dashboard Streamlit.

Pour lancer uniquement le dashboard 
```bash
cd dashboard
streamlit run dashboard.py
```

  Déis techniques couverts

Merise (MCD/MLD), Star Schema, DuckDB, Régression linéaire, Random orest, Comparaison de modèles, Matrice de corrélation, Streamlit.

  Badges visés

Data Engineer (ETL + DuckDB + pipeline reproductible), Data Architect (Merise + Star Schema), IA (scikit-learn, entraînement et évaluation de modèles).
