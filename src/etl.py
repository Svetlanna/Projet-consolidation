import pandas as pd

meteo = pd.read_csv("../data/raw/Q_75_previous-1950-2024_RR-T-Vent.csv", sep=";")


meteo = meteo[meteo["NUM_POSTE"] == 75114001].copy()

meteo["year"] = meteo["AAAAMMJJ"].astype(str).str[:4].astype(int)



douteux = (meteo["QTM"] == 9) & (meteo["QTN"] == 1) & (meteo["QTX"] == 1)
meteo.loc[douteux, "TM"] = (meteo.loc[douteux, "TN"] + meteo.loc[douteux, "TX"]) / 2

print(f"Jours corrigés : {douteux.sum()}")
print(meteo[["year", "TM"]].head())

# ça regroupe les 365 lignes de chaque année en une seule ligne
meteo_annual = meteo.groupby("year")["TM"].agg(["mean", "count"]).reset_index()
meteo_annual = meteo_annual.rename(columns={"mean": "avg_temp_paris", "count": "nb_jours"})

meteo_annual = meteo_annual[
    (meteo_annual["year"] >= 1958) & (meteo_annual["year"] <= 2024)
]
#
print(meteo_annual.shape)
print(meteo_annual["nb_jours"].min(), meteo_annual["nb_jours"].max())
print(meteo_annual.head())



# nettoyer le fichier CO2
co2 = pd.read_csv("../data/raw/climate_merged.csv")


# pour chaque groupe (chaque année) : "crée-moi une colonne
# co2_ppm en prenant la première valeur de
co2_annual = co2.groupby("year").agg(
    co2_ppm=("CO2_ppm", "first"),
    temp_anomaly_global=("Temp_Anomaly_C", "mean")
).reset_index()

print(co2_annual.shape)
print(co2_annual.head())




df_clean = pd.merge(
    co2_annual,
    meteo_annual[["year", "avg_temp_paris"]],
    on="year",
    how="inner"
)

print(df_clean.shape)
print(df_clean.isna().sum())   # vérifie qu'il n'y a aucune valeur manquante après fusion
print(df_clean.head())

df_clean.to_csv("../data/clean/climate_paris_clean.csv", index=False)
print("Fichier sauvegardé.")