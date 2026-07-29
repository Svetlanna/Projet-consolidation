import streamlit as st
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CO2 & Température en France", layout="wide")
st.title("Impact du CO2 sur la température en France")

con = duckdb.connect("../db/climat.duckdb")
df = con.execute("SELECT * FROM fait_climat ORDER BY annee").fetchdf()
con.close()

col1, col2, col3 = st.columns(3)
col1.metric("Dernière année disponible", int(df["annee"].max()))
col2.metric("CO2 (dernière année)", f'{df["co2_ppm"].iloc[-1]:.1f} ppm')
col3.metric("Température moyenne France (dernière année)", f'{df["avg_temp_france"].iloc[-1]:.1f} °C')

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