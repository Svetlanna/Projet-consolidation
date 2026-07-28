import streamlit as st
import duckdb
import pandas as pd

st.set_page_config(page_title="CO2 & Température Paris", layout="wide")
st.title("Impact du CO2 sur la température à Paris")

con = duckdb.connect("../db/climat.duckdb")
df = con.execute("SELECT * FROM fait_climat ORDER BY annee").fetchdf()
con.close()

col1, col2, col3 = st.columns(3)
col1.metric("Dernière année disponible", int(df["annee"].max()))
col2.metric("CO2 (dernière année)", f'{df["co2_ppm"].iloc[-1]:.1f} ppm')
col3.metric("Température moyenne Paris (dernière année)", f'{df["avg_temp_paris"].iloc[-1]:.1f} °C')

st.dataframe(df)