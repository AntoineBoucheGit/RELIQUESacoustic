import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import pingouin as pg
import random
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(22)
random.seed(22)

df = pd.read_csv('/Users/superman/Desktop/Stage/IndicesReliques/ReliquesIndicesTotAENettoyé.csv', sep=',')
df['Distance_lisiere'] = pd.to_numeric(df['Distance_lisiere'].astype(str).str.extract(r'(\d+)')[0])
df = df[df["rainy (1 = rainy, 0 = not rainy)"] == 0]
df["Heure_num"] = df["Heure"].astype(str).str.zfill(6).str[:2].astype(int)

# Intervalles horaires de 3h
df["Intervalle"] = pd.cut(df["Heure_num"],
                          bins=[0, 3, 6, 9, 12, 15, 18, 21,24],
                          labels=["0-3h", "3-6h", "6-9h", "9-12h", "12-15h", "15-18h", "18-21h", "21-00h"],
                          right=False, include_lowest=True)

# Colonnes features
features_cols = df.columns.difference([
    "Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone",
    "Distance_lisiere", "Heure", "Heure_num", "Date",
    "jour/nuit", "MEANt", "Intervalle", "Identifiant"
])

zone_to_distance = df.groupby("Zone")["Distance_lisiere"].agg(lambda x: x.mode()[0]).to_dict()

#  SÉLECTION 

results = []
selected_zones = df["Zone"].unique()  

for zone in selected_zones:
    df_zone = df[df["Zone"] == zone]

    valid_days = []
    for date, group in df_zone.groupby("Date"):
        if set(group["Intervalle"].dropna().unique()) == set(df["Intervalle"].cat.categories):
            valid_days.append(date)

    print(f"Zone {zone} : {len(valid_days)} jours valides") 

    if len(valid_days) < 4:
        print(f"Zone {zone} ignorée (jours valides insuffisants : {len(valid_days)})")
        continue

    n_days = 6 if len(valid_days) >= 6 else 4
    selected_days = np.random.choice(valid_days, size=n_days, replace=False)

    for date in selected_days:
        df_day = df_zone[df_zone["Date"] == date]

        for intervalle, group in df_day.groupby("Intervalle"):
            mean_vals = group[features_cols].mean()
            result = {
                "Zone": zone,
                "Date": date,  
                "Intervalle": intervalle,
                "Distance_lisiere": zone_to_distance[zone]
            }
            result.update(mean_vals.to_dict())
            results.append(result)


# Df result

df_result = pd.DataFrame(results)
df_result = df_result.sort_values(
    by=["Distance_lisiere", "Zone", "Date", "Intervalle"]
).reset_index(drop=True)

if df_result.shape[0] != 1440:
    print(f"\n⚠️ Seulement {df_result.shape[0]} lignes au lieu de 1440. Certains sites ont été ignorés.")

# MATRICE DE DISTANCE EUCLIDIENNE 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_result[features_cols])
df_scaled = df_result.copy()
df_scaled[features_cols] = X_scaled


distance_matrix = squareform(pdist(df_scaled[features_cols], metric='euclidean'))


df_scaled["Label"] = df_scaled.apply(
    lambda row: f"{row['Zone']} | {row['Date']} | {row['Intervalle']}", axis=1
)
labels = df_scaled["Label"]

# VMAX

vmax = np.percentile(distance_matrix, 95)

plt.figure(figsize=(14, 12))

# # Heatmap
ax = sns.heatmap(
    distance_matrix,
    cmap="viridis",
    square=True,
    vmax=vmax,
    cbar_kws={"label": "Distance euclidienne"},
    xticklabels=False,
    yticklabels=False
)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(30)  

plt.tight_layout()

# # Sauvegarde 

plt.savefig("matrice_distance_euclidienne.pdf")

plt.show()


# MATRICE DE DIFFÉRENCE DE DISTANCE À LA LISIÈRE 

lisiere_distance_matrix = np.zeros((len(df_result), len(df_result)))

for i in range(len(df_result)):
    for j in range(len(df_result)):
        lisiere_distance_matrix[i, j] = abs(df_result["Distance_lisiere"].iloc[i] - df_result["Distance_lisiere"].iloc[j]) if i != j else 0





plt.figure(figsize=(10, 8))

# heatmap
ax = sns.heatmap(
    lisiere_distance_matrix,
    cmap="viridis",
    square=True,
    fmt=".0f",
    cbar_kws={"label": "Différence de distance à la lisière (m)"},
    xticklabels=False,
    yticklabels=False
)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(28) 
plt.tight_layout()

# Sauvegarde 

plt.savefig("matrice_distance_lisiere.pdf")

plt.show()


# MATRICE SITE

site_matrix = np.zeros((len(df_result), len(df_result)))

for i in range(len(df_result)):
    for j in range(len(df_result)):
        site_matrix[i, j] = 1 if df_result["Zone"].iloc[i] != df_result["Zone"].iloc[j] else 0

# HEATMAP DE LA MATRICE SITE 


plt.figure(figsize=(10, 8))

ax = sns.heatmap(
    site_matrix,
    cmap="viridis",
    square=True,
    cbar_kws={
        "ticks": [0, 1],
        "label": "Appartenance au même site\n(0 = oui, 1 = non)"
    },
    vmin=0, vmax=1,
    xticklabels=False,
    yticklabels=False
)


cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(30) 

plt.savefig("matrice_site.pdf")
plt.tight_layout()
plt.show()
