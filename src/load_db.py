import os
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "clean", "climate_france_clean.csv")
DB_PATH = os.path.join(BASE_DIR, "db", "climat.duckdb")

df = pd.read_csv(CSV_PATH)
df["decennie"] = (df["year"] // 10) * 10
df["periode"] = pd.cut(
    df["year"],
    bins=[1957, 1979, 1999, 2100],
    labels=["1958-1979", "1980-1999", "2000-2024"],
).astype(str)

classes = pd.DataFrame({
    "id_classe": [1, 2, 3],
    "libelle": ["Année fraîche", "Année normale", "Année chaude"],
    "temp_min": [-100.0, 11.5, 12.5],
    "temp_max": [11.5, 12.5, 100.0],
})


def classe_de(temp):
    for _, c in classes.iterrows():
        if c["temp_min"] <= temp < c["temp_max"]:
            return int(c["id_classe"])
    return None


df["id_classe"] = df["avg_temp_france"].apply(classe_de)

con = duckdb.connect(DB_PATH)

con.execute("DROP TABLE IF EXISTS predictions")
con.execute("DROP TABLE IF EXISTS fait_climat")
con.execute("DROP TABLE IF EXISTS dim_annee")
con.execute("DROP TABLE IF EXISTS dim_classe_thermique")

con.execute("""
    CREATE TABLE dim_annee (
        annee INTEGER PRIMARY KEY,
        decennie INTEGER NOT NULL,
        periode VARCHAR NOT NULL
    )
""")

con.execute("""
    CREATE TABLE dim_classe_thermique (
        id_classe INTEGER PRIMARY KEY,
        libelle VARCHAR NOT NULL,
        temp_min DOUBLE,
        temp_max DOUBLE
    )
""")

con.execute("""
    CREATE TABLE fait_climat (
        annee INTEGER PRIMARY KEY REFERENCES dim_annee(annee),
        id_classe INTEGER REFERENCES dim_classe_thermique(id_classe),
        co2_ppm DOUBLE,
        temp_anomaly_global DOUBLE,
        co2_growth DOUBLE,
        avg_temp_france DOUBLE
    )
""")

con.execute("INSERT INTO dim_annee SELECT year, decennie, periode FROM df")
con.execute("INSERT INTO dim_classe_thermique SELECT * FROM classes")
con.execute("""
    INSERT INTO fait_climat
    SELECT year, id_classe, co2_ppm, temp_anomaly_global, co2_growth, avg_temp_france
    FROM df
""")

print(con.execute("SELECT * FROM fait_climat ORDER BY annee LIMIT 5").fetchdf())
print(con.execute("SELECT COUNT(*) AS nb_lignes FROM fait_climat").fetchdf())
print(con.execute("SELECT * FROM dim_annee ORDER BY annee LIMIT 5").fetchdf())
print(con.execute("SELECT * FROM dim_classe_thermique").fetchdf())
con.close()
