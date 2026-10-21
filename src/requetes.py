import os
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "climat.duckdb")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

REQUETES = {
    "nb_annees_par_classe": """
        SELECT c.libelle AS classe, COUNT(*) AS nb_annees
        FROM fait_climat f
        JOIN dim_classe_thermique c ON f.id_classe = c.id_classe
        GROUP BY c.id_classe, c.libelle
        ORDER BY c.id_classe
    """,

    "moyennes_par_decennie": """
        SELECT a.decennie,
               ROUND(AVG(f.avg_temp_france), 2) AS temp_moyenne_c,
               ROUND(AVG(f.co2_ppm), 1) AS co2_moyen_ppm,
               ROUND(AVG(f.temp_anomaly_global), 3) AS anomalie_mondiale
        FROM fait_climat f
        JOIN dim_annee a ON f.annee = a.annee
        GROUP BY a.decennie
        ORDER BY a.decennie
    """,

    "evolution_par_periode": """
        SELECT a.periode,
               MIN(f.annee) AS debut,
               MAX(f.annee) AS fin,
               ROUND(AVG(f.avg_temp_france), 2) AS temp_moyenne_c,
               ROUND(MAX(f.co2_ppm) - MIN(f.co2_ppm), 1) AS hausse_co2_ppm
        FROM fait_climat f
        JOIN dim_annee a ON f.annee = a.annee
        GROUP BY a.periode
        ORDER BY a.periode
    """,

    "top5_annees_chaudes": """
        SELECT f.annee, ROUND(f.avg_temp_france, 2) AS temp_c, ROUND(f.co2_ppm, 1) AS co2_ppm, c.libelle AS classe
        FROM fait_climat f
        JOIN dim_classe_thermique c ON f.id_classe = c.id_classe
        ORDER BY f.avg_temp_france DESC
        LIMIT 5
    """,

    "ecart_a_la_normale_1961_1990": """
        WITH normale AS (
            SELECT AVG(avg_temp_france) AS t_ref
            FROM fait_climat
            WHERE annee BETWEEN 1961 AND 1990
        )
        SELECT f.annee,
               ROUND(f.avg_temp_france, 2) AS temp_c,
               ROUND(f.avg_temp_france - n.t_ref, 2) AS ecart_normale_c
        FROM fait_climat f, normale n
        ORDER BY f.annee
    """,

    "croissance_co2_par_decennie": """
        SELECT a.decennie,
               ROUND(AVG(f.co2_growth), 2) AS hausse_annuelle_moyenne_ppm,
               ROUND(MAX(f.co2_growth), 2) AS hausse_max_ppm
        FROM fait_climat f
        JOIN dim_annee a ON f.annee = a.annee
        WHERE f.co2_growth IS NOT NULL
        GROUP BY a.decennie
        ORDER BY a.decennie
    """,

    "moyenne_mobile_10_ans": """
        SELECT annee,
               ROUND(avg_temp_france, 2) AS temp_c,
               ROUND(AVG(avg_temp_france) OVER (ORDER BY annee ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 2) AS moyenne_mobile_10_ans
        FROM fait_climat
        ORDER BY annee
    """,
}


def executer(nom, con=None):
    fermer = con is None
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)
    resultat = con.execute(REQUETES[nom]).fetchdf()
    if fermer:
        con.close()
    return resultat


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.set_option("display.width", 160)
    con = duckdb.connect(DB_PATH, read_only=True)
    for nom in REQUETES:
        resultat = executer(nom, con)
        print(f"\n=== {nom} ===")
        print(resultat.head(10).to_string(index=False))
        resultat.to_csv(os.path.join(RESULTS_DIR, f"sql_{nom}.csv"), index=False)
    con.close()
