import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

etapes = [
    ("ETL (nettoyage des données)", "etl.py"),
    ("Chargement dans DuckDB", "load_db.py"),
    ("Analyse + modèle ML", "model.py"),
]

for label, script in etapes:
    print(f"\n=== {label} ===")
    dossier_src = os.path.join(BASE_DIR, "src")
    resultat = subprocess.run([sys.executable, script], cwd=dossier_src)
    if resultat.returncode != 0:
        print(f"Erreur dans {script}, arrêt du pipeline.")
        sys.exit(1)
#




#
print("\n=== Lancement du dashboard Streamlit ===")
dossier_dashboard = os.path.join(BASE_DIR, "dashboard")
subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], cwd=dossier_dashboard)