import streamlit as st
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score



st.set_page_config(page_title="CO2 & Température en France", layout="wide")
st.title("Impact du CO2 sur la température en France")

con = duckdb.connect("../db/climat.duckdb")
df = con.execute("SELECT * FROM fait_climat ORDER BY annee").fetchdf()
con.close()

col1, col2, col3 = st.columns(3)
col1.metric("Dernière année disponible", int(df["annee"].max()))
col2.metric("CO2 (dernière année)", f'{df["co2_ppm"].iloc[-1]:.1f} ppm')
col3.metric("Température moyenne France (dernière année)", f'{df["avg_temp_france"].iloc[-1]:.1f} °C')





#
st.subheader("Performance du modèle prédictif")

X = df[["co2_ppm"]]
y = df["avg_temp_france"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col_m1, col_m2 = st.columns(2)
col_m1.metric("Erreur moyenne (MAE)", f"{mae:.3f} °C")
col_m2.metric("Variance expliquée (R²)", f"{r2:.3f}")

co2_trend = LinearRegression().fit(df[["annee"]], df["co2_ppm"])
co2_2035 = co2_trend.predict(pd.DataFrame({"annee": [2035]}))[0]
temp_2035 = lin_model.predict(pd.DataFrame({"co2_ppm": [co2_2035]}))[0]

st.subheader("Projection 2035")
col_p1, col_p2 = st.columns(2)
col_p1.metric("CO2 projeté", f"{co2_2035:.1f} ppm")
col_p2.metric("Température projetée", f"{temp_2035:.2f} °C")
#




st.dataframe(df)

st.subheader("Évolution du CO2 et de la température (1958-2024)")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["annee"], df["co2_ppm"], color="tab:blue")
ax1.set_xlabel("Année")
ax1.set_ylabel("CO2 (ppm)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(df["annee"], df["avg_temp_france"], color="tab:red")
ax2.set_ylabel("Température (°C)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle("CO2 et température évoluent ensemble dans le temps")
fig.tight_layout()
st.pyplot(fig)

st.subheader("Matrice de corrélation")
correlation = df[["co2_ppm", "temp_anomaly_global", "co2_growth", "avg_temp_france"]].corr()




fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
st.pyplot(fig)