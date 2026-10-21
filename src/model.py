import os
import time
import inspect
import duckdb
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "climat.duckdb")
MODEL_PATH = os.path.join(BASE_DIR, "db", "temp_predictor.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_PATH = os.path.join(BASE_DIR, "data", "clean", "correlation_matrix.png")

FEATURES = ["co2_ppm"]
CIBLE = "avg_temp_france"
TOLERANCE = 0.5
ANNEE_COUPURE = 2011


def charger_donnees():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT f.annee, f.co2_ppm, f.temp_anomaly_global, f.co2_growth, f.avg_temp_france
        FROM fait_climat f
        JOIN dim_annee a ON f.annee = a.annee
        ORDER BY f.annee
    """).fetchdf()
    con.close()
    return df


def decouper(df):
    train = df[df["annee"] < ANNEE_COUPURE]
    test = df[df["annee"] >= ANNEE_COUPURE]
    return train[FEATURES], test[FEATURES], train[CIBLE], test[CIBLE], test["annee"]


def construire_modeles():
    return {
        "Regression lineaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "Reseau de neurones MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=5000, learning_rate_init=0.01, random_state=42),
        ),
    }


def evaluer(y_reel, y_pred):
    return {
        "MAE": mean_absolute_error(y_reel, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_reel, y_pred))),
        "R2": r2_score(y_reel, y_pred),
        "Accuracy_0.5C": float(np.mean(np.abs(np.asarray(y_reel) - np.asarray(y_pred)) <= TOLERANCE)),
    }


def nb_hyperparametres(modele):
    estimateur = modele.steps[-1][1] if hasattr(modele, "steps") else modele
    return len(inspect.signature(type(estimateur).__init__).parameters) - 1


def sauvegarder_predictions(resultats):
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            annee INTEGER REFERENCES dim_annee(annee),
            modele VARCHAR,
            prediction DOUBLE,
            reel DOUBLE,
            erreur_abs DOUBLE,
            correct BOOLEAN
        )
    """)
    con.execute("INSERT INTO predictions SELECT * FROM resultats")
    erreurs = con.execute(f"""
        SELECT annee, modele, ROUND(prediction, 2) AS prediction, ROUND(reel, 2) AS reel, ROUND(erreur_abs, 2) AS erreur
        FROM predictions
        WHERE ABS(prediction - reel) > {TOLERANCE}
        ORDER BY modele, annee
    """).fetchdf()
    nb_erreurs = con.execute(f"""
        SELECT modele, COUNT(*) AS nb_predictions,
               COUNT(*) FILTER (WHERE ABS(prediction - reel) > {TOLERANCE}) AS nb_erreurs
        FROM predictions
        GROUP BY modele
        ORDER BY nb_erreurs
    """).fetchdf()
    con.close()
    return erreurs, nb_erreurs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = charger_donnees()
    print(df.shape)

    correlation = df[["co2_ppm", "temp_anomaly_global", "co2_growth", "avg_temp_france"]].corr()
    print(correlation.round(2))

    X_train, X_test, y_train, y_test, annees_test = decouper(df)
    print(f"Entraînement : {len(X_train)} années (< {ANNEE_COUPURE}), test : {len(X_test)} années (>= {ANNEE_COUPURE})")

    lignes = []
    comparaison = []
    modeles = construire_modeles()
    for nom, modele in modeles.items():
        durees = []
        for _ in range(3):
            debut = time.perf_counter()
            modele.fit(X_train, y_train)
            durees.append(time.perf_counter() - debut)
        duree = min(durees)
        pred = modele.predict(X_test)
        scores = evaluer(y_test, pred)
        comparaison.append({
            "Modele": nom,
            **{k: round(v, 3) for k, v in scores.items()},
            "Temps_entrainement_s": round(duree, 4),
            "Hyperparametres_reglables": nb_hyperparametres(modele),
        })
        for annee, p, r in zip(annees_test, pred, y_test):
            lignes.append({
                "annee": int(annee),
                "modele": nom,
                "prediction": float(p),
                "reel": float(r),
                "erreur_abs": float(abs(p - r)),
                "correct": bool(abs(p - r) <= TOLERANCE),
            })

    tableau = pd.DataFrame(comparaison)
    tableau["Complexite_code"] = ["Faible", "Faible", "Moyenne"]
    tableau["Facilite_reglage"] = ["Très facile", "Facile", "Difficile"]
    print("\n=== Comparaison des modèles ===")
    print(tableau.to_string(index=False))
    tableau.to_csv(os.path.join(RESULTS_DIR, "comparaison_modeles.csv"), index=False)

    resultats = pd.DataFrame(lignes)
    resultats.insert(0, "id", range(1, len(resultats) + 1))
    reel_vs_predit = resultats.pivot(index="annee", columns="modele", values="prediction")
    reel_vs_predit.insert(0, "Reel", resultats.groupby("annee")["reel"].first())
    print("\n=== Réel vs prédit ===")
    print(reel_vs_predit.round(2).to_string())
    reel_vs_predit.round(3).to_csv(os.path.join(RESULTS_DIR, "reel_vs_predit.csv"))

    erreurs, nb_erreurs = sauvegarder_predictions(resultats)
    print("\n=== Prédictions avec plus de 0.5 °C d'erreur ===")
    print(erreurs.to_string(index=False))
    print("\n=== Nombre d'erreurs par modèle ===")
    print(nb_erreurs.to_string(index=False))

    meilleur_nom = tableau.sort_values("MAE").iloc[0]["Modele"]
    meilleur = construire_modeles()[meilleur_nom].fit(df[FEATURES], df[CIBLE])
    joblib.dump({"nom": meilleur_nom, "modele": meilleur, "features": FEATURES}, MODEL_PATH)
    print(f"\nMeilleur modèle : {meilleur_nom}, réentraîné sur toutes les années et sauvegardé")

    co2_trend = LinearRegression().fit(df[["annee"]], df["co2_ppm"])
    co2_2035 = co2_trend.predict(pd.DataFrame({"annee": [2035]}))[0]
    temp_2035 = meilleur.predict(pd.DataFrame({"co2_ppm": [co2_2035]}))[0]
    print(f"CO2 projeté en 2035 : {co2_2035:.1f} ppm")
    print(f"Température moyenne projetée à Paris en 2035 : {temp_2035:.2f} °C")
    pd.DataFrame([{"annee": 2035, "co2_ppm": round(co2_2035, 1), "temp_c": round(temp_2035, 2), "modele": meilleur_nom}]).to_csv(
        os.path.join(RESULTS_DIR, "projection_2035.csv"), index=False
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Matrice de corrélation — CO2 et température")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
