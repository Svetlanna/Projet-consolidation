import pandas as pd


meteo = pd.read_csv("../data/raw/Q_75_previous-1950-2024_RR-T-Vent.csv", sep=";")
meteo["year"] = meteo["AAAAMMJJ"].astype(str).str[:4].astype(int)


douteux = (meteo["QTM"] == 9) & (meteo["QTN"] == 1) & (meteo["QTX"] == 1)
meteo.loc[douteux, "TM"] = (meteo.loc[douteux, "TN"] + meteo.loc[douteux, "TX"]) / 2

print(f"Jours corrigés : {douteux.sum()}")
print(meteo[["year", "TM"]].head())


meteo_annual = meteo.groupby("year")["TM"].agg(["mean", "count"]).reset_index()
meteo_annual = meteo_annual.rename(columns={"mean": "avg_temp_france", "count": "nb_mesures"})
meteo_annual = meteo_annual[(meteo_annual["year"] >= 1958) & (meteo_annual["year"] <= 2024)]

print(meteo_annual.shape)
print(meteo_annual[["year", "nb_mesures"]])

# --- Partie CO2 / température mondiale ---
co2 = pd.read_csv("../data/raw/climate_merged.csv")

co2_annual = co2.groupby("year").agg(
    co2_ppm=("CO2_ppm", "first"),
    temp_anomaly_global=("Temp_Anomaly_C", "mean")
).reset_index()
co2_annual["co2_growth"] = co2_annual["co2_ppm"].diff()
co2_annual["temp_5yr_ma"] = co2_annual["temp_anomaly_global"].rolling(window=5).mean()

print(co2_annual.shape)
print(co2_annual.head())

# --- Fusion des deux sources ---
df_clean = pd.merge(
    co2_annual,
    meteo_annual[["year", "avg_temp_france"]],
    on="year",
    how="inner"
)








#
#
# meteo_par_station = meteo.groupby(["year", "NUM_POSTE", "NOM_USUEL"])["TM"].mean().reset_index()
#
# meteo_par_station = meteo_par_station.rename(columns={"TM": "avg_temp"})
# meteo_par_station = meteo_par_station[
#     (meteo_par_station["year"] >= 1958) & (meteo_par_station["year"] <= 2024)
# ]
#
# detail = pd.merge(meteo_par_station, co2_annual, on="year", how="inner")
#
# pd.set_option("display.max_rows", None)
# print(detail[["NUM_POSTE", "year", "co2_ppm", "temp_anomaly_global", "co2_growth", "avg_temp"]])
#
# detail.to_csv("../data/clean/climate_par_station.csv", index=False)




print(df_clean.shape)
print(df_clean.isna().sum())
print(df_clean.head())

df_clean.to_csv("../data/clean/climate_france_clean.csv", index=False)
print("Fichier sauvegardé.")