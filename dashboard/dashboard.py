import os
import sys
import duckdb
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.join(BASE_DIR, "..")
DB_PATH = os.path.join(RACINE, "db", "climat.duckdb")
MODEL_PATH = os.path.join(RACINE, "db", "temp_predictor.pkl")
RESULTS_DIR = os.path.join(RACINE, "results")
sys.path.append(os.path.join(RACINE, "src"))

from requetes import REQUETES

st.set_page_config(page_title="CO2 & Température en France", layout="wide")


@st.cache_data
def lire_sql(sql):
    con = duckdb.connect(DB_PATH, read_only=True)
    resultat = con.execute(sql).fetchdf()
    con.close()
    return resultat


@st.cache_data
def lire_csv(nom):
    chemin = os.path.join(RESULTS_DIR, nom)
    return pd.read_csv(chemin) if os.path.exists(chemin) else None


@st.cache_resource
def charger_modele():
    objet = joblib.load(MODEL_PATH)
    if isinstance(objet, dict):
        return objet["nom"], objet["modele"]
    return "Regression lineaire", objet


df = lire_sql("""
    SELECT f.annee, a.decennie, a.periode, c.libelle AS classe,
           f.co2_ppm, f.temp_anomaly_global, f.co2_growth, f.avg_temp_france
    FROM fait_climat f
    JOIN dim_annee a ON f.annee = a.annee
    JOIN dim_classe_thermique c ON f.id_classe = c.id_classe
    ORDER BY f.annee
""")

st.title("Impact du CO2 sur la température en France")

onglet_pres, onglet_explo, onglet_sql, onglet_ia, onglet_mlops = st.tabs(
    ["Présentation", "Exploration", "Requêtes SQL", "Intelligence artificielle", "MLOps"]
)

with onglet_pres:
    st.header("Le projet")
    st.write(
        "Ce projet relie deux sources de données ouvertes pour mesurer, sur des données réelles, "
        "le lien entre la concentration de CO2 dans l'atmosphère et la température à Paris"
    )
    st.header("La problématique")
    st.info(
        "Comment le CO2 influence-t-il les températures à Paris, et peut-on prédire la température "
        "moyenne annuelle à l'horizon 2035 à partir de la concentration mondiale de CO2 ?"
    )
    st.header("Les données")
    st.markdown(
        "**Kaggle, Global CO2 vs Temp Anomaly** : concentration de CO2 (ppm) et anomalie de température mondiale, 1958 à 2024  \n"
        "**Météo France (data.gouv.fr)** : températures quotidiennes de toutes les stations du département 75, moyennées par année  \n"
        "**Base DuckDB en étoile** : table de faits `fait_climat`, dimensions `dim_annee` et `dim_classe_thermique`"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Années étudiées", len(df))
    k2.metric("CO2 en 2024", f'{df["co2_ppm"].iloc[-1]:.1f} ppm', f'+{df["co2_ppm"].iloc[-1] - df["co2_ppm"].iloc[0]:.1f} depuis 1958')
    debut = df[df["annee"] <= 1967]["avg_temp_france"].mean()
    fin = df[df["annee"] >= 2015]["avg_temp_france"].mean()
    k3.metric("Température 2015 à 2024", f"{fin:.2f} °C", f"{fin - debut:+.2f} °C vs 1958 à 1967")
    k4.metric("Corrélation CO2 / température", f'{df["co2_ppm"].corr(df["avg_temp_france"]):.2f}')

with onglet_explo:
    st.header("Explorer les données")
    f1, f2, f3 = st.columns(3)
    plage = f1.slider("Années", int(df["annee"].min()), int(df["annee"].max()), (int(df["annee"].min()), int(df["annee"].max())))
    periodes = f2.multiselect("Périodes", sorted(df["periode"].unique()), default=sorted(df["periode"].unique()))
    classes = ["Toutes"] + list(df["classe"].unique())
    classe = f3.selectbox("Classe thermique", classes)

    vue = df[(df["annee"].between(*plage)) & (df["periode"].isin(periodes))]
    if classe != "Toutes":
        vue = vue[vue["classe"] == classe]

    if vue.empty:
        st.warning("Aucune année ne correspond à ces filtres")
    else:
        st.caption(f"{len(vue)} années sélectionnées")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=vue["annee"], y=vue["co2_ppm"], name="CO2 (ppm)", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=vue["annee"], y=vue["avg_temp_france"], name="Température (°C)", mode="lines+markers"), secondary_y=True)
        fig.update_layout(title="Évolution du CO2 et de la température", hovermode="x unified")
        fig.update_yaxes(title_text="CO2 (ppm)", secondary_y=False)
        fig.update_yaxes(title_text="Température (°C)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Ce que montre ce graphique** : le CO2 monte de façon régulière, la température monte aussi mais en zigzag, car chaque année a sa propre météo")

        fig2 = px.scatter(vue, x="co2_ppm", y="avg_temp_france", color="periode", hover_data=["annee"],
                          labels={"co2_ppm": "CO2 (ppm)", "avg_temp_france": "Température (°C)", "periode": "Période"},
                          title="Température en fonction du CO2")
        if len(vue) > 2:
            droite = LinearRegression().fit(vue[["co2_ppm"]], vue["avg_temp_france"])
            xs = pd.DataFrame({"co2_ppm": [vue["co2_ppm"].min(), vue["co2_ppm"].max()]})
            fig2.add_trace(go.Scatter(x=xs["co2_ppm"], y=droite.predict(xs), mode="lines", name="Tendance", line=dict(color="black", dash="dash")))
        st.plotly_chart(fig2, width="stretch")
        st.markdown("**Ce que montre ce graphique** : chaque point est une année, plus le CO2 est élevé plus les points sont hauts, la droite résume cette tendance")

        par_decennie = vue.groupby("decennie", as_index=False)["avg_temp_france"].mean()
        fig3 = px.bar(par_decennie, x="decennie", y="avg_temp_france", text_auto=".2f",
                      labels={"decennie": "Décennie", "avg_temp_france": "Température moyenne (°C)"},
                      title="Température moyenne par décennie")
        fig3.update_yaxes(range=[10, 14])
        st.plotly_chart(fig3, width="stretch")
        st.markdown("**Ce que montre ce graphique** : en lissant sur dix ans, on voit que les années 2020 sont les plus chaudes de la série")

        corr = vue[["co2_ppm", "temp_anomaly_global", "co2_growth", "avg_temp_france"]].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matrice de corrélation")
        st.plotly_chart(fig4, width="stretch")
        st.markdown("**Ce que montre ce graphique** : une valeur proche de 1 indique que deux variables montent ensemble, CO2 et température de Paris sont corrélés à 0,64 sur toute la période")

        with st.expander("Voir les données filtrées"):
            st.dataframe(vue, width="stretch")

with onglet_sql:
    st.header("Requêtes SQL sur le schéma en étoile")
    nom = st.selectbox("Choisissez une requête", list(REQUETES.keys()))
    st.code(REQUETES[nom].strip(), language="sql")
    st.dataframe(lire_sql(REQUETES[nom]), width="stretch")

with onglet_ia:
    st.header("Prédire la température à partir du CO2")
    nom_modele, modele = charger_modele()

    comparaison = lire_csv("comparaison_modeles.csv")
    if comparaison is not None:
        st.subheader("Comparaison des modèles")
        st.caption("Entraînement sur 1958 à 2010, test sur 2011 à 2024, une prédiction est jugée juste si l'erreur est au plus de 0,5 °C")
        st.dataframe(comparaison, width="stretch", hide_index=True)

    reel = lire_csv("reel_vs_predit.csv")
    if reel is not None:
        st.subheader("Valeurs réelles et valeurs prédites")
        longue = reel.melt(id_vars="annee", var_name="Série", value_name="Température (°C)")
        fig5 = px.line(longue, x="annee", y="Température (°C)", color="Série", markers=True)
        st.plotly_chart(fig5, width="stretch")
        st.markdown(
            "**Ce que montre ce graphique** : la Random Forest prédit la même valeur pour toutes les années récentes, "
            "car un arbre ne sait pas extrapoler au delà des valeurs vues à l'entraînement, le réseau de neurones surestime, "
            "la régression linéaire suit le mieux la tendance"
        )

    st.subheader("Analyse des erreurs enregistrées en SQL")
    seuil = st.slider("Seuil d'erreur (°C)", 0.1, 2.0, 0.5, 0.1)
    erreurs = lire_sql(f"""
        SELECT modele, COUNT(*) AS nb_predictions,
               COUNT(*) FILTER (WHERE ABS(prediction - reel) > {seuil}) AS nb_erreurs,
               ROUND(AVG(erreur_abs), 3) AS erreur_moyenne
        FROM predictions
        GROUP BY modele
        ORDER BY nb_erreurs
    """)
    st.dataframe(erreurs, width="stretch", hide_index=True)

    st.subheader(f"Faire une prédiction ({nom_modele})")
    c1, c2 = st.columns(2)
    co2_choisi = c1.number_input("Concentration de CO2 (ppm)", 280.0, 600.0, 430.0, 1.0)
    temp_pred = modele.predict(pd.DataFrame({"co2_ppm": [co2_choisi]}))[0]
    c1.metric("Température moyenne prédite", f"{temp_pred:.2f} °C")

    annee_cible = c2.slider("Projection pour l'année", 2025, 2060, 2035)
    tendance_co2 = LinearRegression().fit(df[["annee"]], df["co2_ppm"])
    co2_proj = tendance_co2.predict(pd.DataFrame({"annee": [annee_cible]}))[0]
    temp_proj = modele.predict(pd.DataFrame({"co2_ppm": [co2_proj]}))[0]
    c2.metric(f"CO2 projeté en {annee_cible}", f"{co2_proj:.1f} ppm")
    c2.metric(f"Température projetée en {annee_cible}", f"{temp_proj:.2f} °C")
    st.caption("Limite : le CO2 est prolongé en ligne droite, alors que les émissions réelles accélèrent, la projection est donc prudente")

with onglet_mlops:
    st.header("Suivi des expériences avec MLflow")
    experiences = lire_csv("experiences_mlflow.csv")
    if experiences is None:
        st.warning("Lancez d'abord src/experiences_mlflow.py")
    else:
        st.dataframe(experiences, width="stretch", hide_index=True)
        fig6 = px.bar(experiences, x="run", y="MAE", text_auto=".3f", title="Erreur moyenne (MAE) par expérience, plus bas est meilleur")
        st.plotly_chart(fig6, width="stretch")
        st.markdown(
            "**Ce que l'on observe** : augmenter le nombre d'arbres ou leur profondeur n'améliore pas la Random Forest, "
            "elle reste incapable de prédire au delà des températures déjà vues, la régression linéaire reste la meilleure, "
            "et une régularisation Ridge trop forte aplatit la droite et dégrade la MAE"
        )
        st.caption("Pour l'interface complète : mlflow ui --backend-store-uri sqlite:///mlflow.db")
