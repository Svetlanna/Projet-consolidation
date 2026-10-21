import os
import sys
import duckdb
import joblib
import pandas as pd
import pytest

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "src"))

from requetes import REQUETES
from model import evaluer

DB_PATH = os.path.join(RACINE, "db", "climat.duckdb")
MODEL_PATH = os.path.join(RACINE, "db", "temp_predictor.pkl")
CSV_PATH = os.path.join(RACINE, "data", "clean", "climate_france_clean.csv")


@pytest.fixture(scope="module")
def con():
    connexion = duckdb.connect(DB_PATH, read_only=True)
    yield connexion
    connexion.close()


def test_csv_nettoye():
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 67
    assert df["year"].min() == 1958 and df["year"].max() == 2024
    assert df["avg_temp_france"].isna().sum() == 0
    assert df["year"].is_unique


def test_tables_du_schema_en_etoile(con):
    tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    assert {"fait_climat", "dim_annee", "dim_classe_thermique", "predictions"} <= tables


def test_cles_etrangeres_respectees(con):
    orphelins_annee = con.execute("""
        SELECT COUNT(*) FROM fait_climat f
        LEFT JOIN dim_annee a ON f.annee = a.annee
        WHERE a.annee IS NULL
    """).fetchone()[0]
    orphelins_classe = con.execute("""
        SELECT COUNT(*) FROM fait_climat f
        LEFT JOIN dim_classe_thermique c ON f.id_classe = c.id_classe
        WHERE c.id_classe IS NULL
    """).fetchone()[0]
    assert orphelins_annee == 0
    assert orphelins_classe == 0


def test_valeurs_plausibles(con):
    tmin, tmax, cmin, cmax = con.execute("""
        SELECT MIN(avg_temp_france), MAX(avg_temp_france), MIN(co2_ppm), MAX(co2_ppm) FROM fait_climat
    """).fetchone()
    assert 8 < tmin < tmax < 16
    assert 300 < cmin < cmax < 450


@pytest.mark.parametrize("nom", list(REQUETES.keys()))
def test_chaque_requete_renvoie_des_lignes(con, nom):
    resultat = con.execute(REQUETES[nom]).fetchdf()
    assert len(resultat) > 0


def test_au_moins_cinq_requetes():
    assert len(REQUETES) >= 5


def test_table_predictions(con):
    nb, nb_modeles = con.execute("SELECT COUNT(*), COUNT(DISTINCT modele) FROM predictions").fetchone()
    assert nb_modeles >= 3
    assert nb == 14 * nb_modeles
    incoherences = con.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE ABS(ABS(prediction - reel) - erreur_abs) > 1e-9
    """).fetchone()[0]
    assert incoherences == 0


def test_modele_sauvegarde_predit_une_temperature_realiste():
    objet = joblib.load(MODEL_PATH)
    modele = objet["modele"]
    pred = modele.predict(pd.DataFrame({"co2_ppm": [320.0, 420.0]}))
    assert 9 < pred[0] < 15 and 9 < pred[1] < 16
    assert pred[1] > pred[0]


def test_metriques_evaluer():
    scores = evaluer([10.0, 12.0, 14.0], [10.0, 12.4, 15.0])
    assert scores["MAE"] == pytest.approx(1.4 / 3)
    assert scores["Accuracy_0.5C"] == pytest.approx(2 / 3)
