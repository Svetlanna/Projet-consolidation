import duckdb
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


con = duckdb.connect("../db/climat.duckdb")
df = con.execute("""
    SELECT f.annee, f.co2_ppm, f.temp_anomaly_global,f.co2_growth, f.avg_temp_france
    FROM fait_climat f
    ORDER BY f.annee
""").fetchdf()
con.close()

print(df.shape)


correlation = df[["co2_ppm", "temp_anomaly_global","co2_growth", "avg_temp_france"]].corr()
print(correlation)









X = df[["co2_ppm"]]
y = df["avg_temp_france"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print(f"Régression linéaire — MAE: {mae_lin:.3f} °C, R²: {r2_lin:.3f}")

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


print(f"Random Forest — MAE: {mae_rf:.3f} °C, R²: {r2_rf:.3f}")



joblib.dump(lin_model, "../db/temp_predictor.pkl")


co2_trend = LinearRegression().fit(df[["annee"]], df["co2_ppm"])
co2_2035 = co2_trend.predict(pd.DataFrame({"annee": [2035]}))[0]
print(f"CO2 projeté en 2035 : {co2_2035:.1f} ppm")





temp_2035 = lin_model.predict(pd.DataFrame({"co2_ppm": [co2_2035]}))[0]
print(f"Température moyenne projetée en France en 2035 : {temp_2035:.2f} °C")


plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matrice de corrélation — CO2 et température")
plt.tight_layout()
plt.savefig("../data/clean/correlation_matrix.png", dpi=150, bbox_inches="tight")





