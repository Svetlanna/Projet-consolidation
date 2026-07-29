import duckdb
import pandas as pd

df = pd.read_csv("../data/clean/climate_france_clean.csv")


df["decennie"] = (df["year"] // 10) * 10

con = duckdb.connect("../db/climat.duckdb")

con.execute("""
    CREATE OR REPLACE TABLE dim_annee AS
    SELECT year AS annee, decennie FROM df
""")

con.execute("""
    CREATE OR REPLACE TABLE fait_climat AS
    SELECT year AS annee, co2_ppm, temp_anomaly_global,co2_growth, avg_temp_france FROM df
""")

# Vérification : on relit ce qu'on vient d'écrire
print(con.execute("SELECT * FROM fait_climat ORDER BY annee LIMIT 5").fetchdf())
print(con.execute("SELECT COUNT(*) FROM fait_climat").fetchdf())
print(con.execute("SELECT * FROM dim_annee ORDER BY annee LIMIT 5").fetchdf())
con.close()



