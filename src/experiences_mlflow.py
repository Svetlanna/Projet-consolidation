import os
import time
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from model import charger_donnees, decouper, evaluer, BASE_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "mlruns")
NOM_EXPERIENCE = "Climat_CO2_Temperature"

EXPERIENCES = [
    {"nom_run": "rf_50_arbres_prof_3", "type": "RandomForest", "params": {"n_estimators": 50, "max_depth": 3, "random_state": 42}},
    {"nom_run": "rf_100_arbres_prof_5", "type": "RandomForest", "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}},
    {"nom_run": "rf_200_arbres_sans_limite", "type": "RandomForest", "params": {"n_estimators": 200, "max_depth": None, "random_state": 42}},
    {"nom_run": "regression_lineaire", "type": "LinearRegression", "params": {"fit_intercept": True}},
    {"nom_run": "ridge_alpha_10000", "type": "Ridge", "params": {"alpha": 10000.0}},
]

CLASSES = {"RandomForest": RandomForestRegressor, "LinearRegression": LinearRegression, "Ridge": Ridge}


def preparer_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(NOM_EXPERIENCE) is None:
        mlflow.create_experiment(NOM_EXPERIENCE, artifact_location="file:///" + ARTIFACTS_DIR.replace("\\", "/").lstrip("/"))
    mlflow.set_experiment(NOM_EXPERIENCE)


def lancer_experiences():
    preparer_mlflow()
    df = charger_donnees()
    X_train, X_test, y_train, y_test, _ = decouper(df)
    lignes = []
    for exp in EXPERIENCES:
        modele = CLASSES[exp["type"]](**exp["params"])
        with mlflow.start_run(run_name=exp["nom_run"]):
            debut = time.perf_counter()
            modele.fit(X_train, y_train)
            duree = time.perf_counter() - debut
            pred = modele.predict(X_test)
            scores = evaluer(y_test, pred)

            mlflow.log_param("type_modele", exp["type"])
            mlflow.log_params({k: str(v) for k, v in exp["params"].items()})
            mlflow.log_param("annee_coupure", 2011)
            mlflow.log_metric("accuracy_0_5C", scores["Accuracy_0.5C"])
            mlflow.log_metric("mae", scores["MAE"])
            mlflow.log_metric("rmse", scores["RMSE"])
            mlflow.log_metric("r2", scores["R2"])
            mlflow.log_metric("temps_entrainement_s", duree)
            mlflow.sklearn.log_model(
                modele,
                name="modele",
                signature=infer_signature(X_test, pred),
                input_example=X_test.head(2),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        lignes.append({"run": exp["nom_run"], **{k: round(v, 3) for k, v in scores.items()}, "temps_s": round(duree, 4)})
        print(f"{exp['nom_run']:<28} MAE={scores['MAE']:.3f}  R2={scores['R2']:.3f}  accuracy={scores['Accuracy_0.5C']:.3f}")

    resultats = pd.DataFrame(lignes).sort_values("MAE")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    resultats.to_csv(os.path.join(RESULTS_DIR, "experiences_mlflow.csv"), index=False)
    print("\n" + resultats.to_string(index=False))
    return resultats


if __name__ == "__main__":
    lancer_experiences()
