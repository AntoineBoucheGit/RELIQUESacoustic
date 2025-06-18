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

df = pd.read_csv('globaldataset.csv', sep=',')
df['Distance_lisiere'] = pd.to_numeric(df['Distance_lisiere'].astype(str).str.extract(r'(\d+)')[0])

# remove files with rain
df = df[df["rainy (1 = rainy, 0 = not rainy)"] == 0]


df["Heure_num"] = df["Heure"].astype(str).str.zfill(6).str[:2].astype(int)

# 3-hour intervals
df["Intervalle"] = pd.cut(df["Heure_num"],
                          bins=[0, 3, 6, 9, 12, 15, 18, 21,24],
                          labels=["0-3h", "3-6h", "6-9h", "9-12h", "12-15h", "15-18h", "18-21h", "21-00h"],
                          right=False, include_lowest=True)

# features
# remove columns thar are NOT features

features_cols = df.columns.difference([
    "Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone",
    "Distance_lisiere", "Heure", "Heure_num", "Date",
    "jour/nuit", "MEANt", "Intervalle", "Identifiant"
])

zone_to_distance = df.groupby("Zone")["Distance_lisiere"].agg(lambda x: x.mode()[0]).to_dict()

#  Selection
# Select only 6 days per site with 8 different time intervals for each day to calculate dissimilarity matrices

results = []

# Unique sites
selected_zones = df["Zone"].unique()  

for zone in selected_zones:
    df_zone = df[df["Zone"] == zone]

# A valid day is a day containing rain-free files in all time intervals
    valid_days = []
    for date, group in df_zone.groupby("Date"):
        if set(group["Intervalle"].dropna().unique()) == set(df["Intervalle"].cat.categories):
            valid_days.append(date)

    if len(valid_days) < 6:
        print(f"Zone {zone} ignored (insufficient valid days : {len(valid_days)})")
        continue

    n_days = 6 if len(valid_days) >= 6 
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

if df_result.shape[0] != 1440: # Number of sites x 6 days x 8 time intervals
    print(f"\n Only {df_result.shape[0]} lines instead of 1440. Some sites have been ignored.")

# ACOUSTIC DISSIMILARITY MATRIX

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_result[features_cols])
df_scaled = df_result.copy()
df_scaled[features_cols] = X_scaled


acoustic_matrix = squareform(pdist(df_scaled[features_cols], metric='euclidean'))


df_scaled["Label"] = df_scaled.apply(
    lambda row: f"{row['Zone']} | {row['Date']} | {row['Intervalle']}", axis=1
)
labels = df_scaled["Label"]

# Heatmap 

 vmax = np.percentile(acoustic_matrix, 100) # adjust if outliers

 plt.figure(figsize=(14, 12))

 ax = sns.heatmap(
     acoustic_matrix,
     cmap="viridis",
     square=True,
     vmax=vmax,
     cbar_kws={"label": "Euclidean distance"},
     xticklabels=False,
     yticklabels=False
 )
 cbar = ax.collections[0].colorbar
 cbar.ax.yaxis.label.set_size(30)  


 plt.tight_layout()
 plt.show()


# EDGE DISTANCE DIFFERENCE MATRIX

lisiere_distance_matrix = np.zeros((len(df_result), len(df_result)))

for i in range(len(df_result)):
    for j in range(len(df_result)):
        lisiere_distance_matrix[i, j] = abs(df_result["Distance_lisiere"].iloc[i] - df_result["Distance_lisiere"].iloc[j]) if i != j else 0



# Heatmap

 plt.figure(figsize=(10, 8))


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
 plt.show()


# SITE MATRIX

site_matrix = np.zeros((len(df_result), len(df_result)))

for i in range(len(df_result)):
    for j in range(len(df_result)):
        site_matrix[i, j] = 1 if df_result["Zone"].iloc[i] != df_result["Zone"].iloc[j] else 0

# Heatmap 



 plt.figure(figsize=(10, 8))

 ax = sns.heatmap(
     site_matrix,
     cmap="viridis",
     square=True,
     cbar_kws={
         "ticks": [0, 1],
         "label": "Belonging to the same site (0 = yes, 1 = no)"
     },
     vmin=0, vmax=1,
     xticklabels=False,
     yticklabels=False
 )


 cbar = ax.collections[0].colorbar
 cbar.ax.yaxis.label.set_size(30) 


 plt.tight_layout()
 plt.show()



# PARTIAL CORRELATION

triu_idx = np.triu_indices_from(acoustic_matrix, k=1)

# Vectors for each matrix

vec_acoustic = acoustic_matrix[triu_idx]

vec_lisiere = lisiere_distance_matrix[triu_idx]

vec_site = site_matrix[triu_idx]


# DataFrame corr
df_corr = pd.DataFrame({
    'acoustic': vec_acoustic,
    'lisiere': vec_lisiere,
    'site': vec_site
})

# Partial correlations


pcorr_G = pg.partial_corr(data=df_corr, x='acoustic', y='lisiere', covar='site', method='spearman')


print("Partial correlation between acoustic matrix and distance to edge matrix (site control):")
print(pcorr_G)


#REPETITION 100 FOIS :

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import pingouin as pg
import random
from tqdm import tqdm  


df = pd.read_csv('/Users/superman/Desktop/Stage/IndicesReliques/ReliquesIndicesTotAENettoyé.csv', sep=',')
df['Distance_lisiere'] = pd.to_numeric(df['Distance_lisiere'].astype(str).str.extract(r'(\d+)')[0])
df = df[df["rainy (1 = rainy, 0 = not rainy)"] == 0]
df["Heure_num"] = df["Heure"].astype(str).str.zfill(6).str[:2].astype(int)

df["Intervalle"] = pd.cut(df["Heure_num"],
                          bins=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                          labels=["0-3h", "3-6h", "6-9h", "9-12h", "12-15h", "15-18h", "18-21h", "21-00h"],
                          right=False, include_lowest=True)

features_cols = df.columns.difference([
    "Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone",
    "Distance_lisiere", "Heure", "Heure_num", "Date",
    "jour/nuit", "MEANt", "Intervalle", "Identifiant"
])
# Liste des features negatives
# A ajouter a features_cols pourr evaluer corr sans indices rajoutant du bruit
features_neg = [
    "ACI", "ACTspCount", "ACTspFract", "ACTspMean", "ACTtCount", "ACTtFraction", "ACTtMean", "AGI",
    "AnthroEnergy", "EAS", "ECU", "ECV", "EPS_KURT", "EPS_SKEW", "EVNspCount", "EVNspFract",
    "EVNtFraction", "EVNtMean", "HFC", "Ht", "KURTt", "LFC", "MFC", "ROU", "SKEWt",
    "SNRf", "SNRt", "TFSD", "VARf", "VARt", "ZCR"
]


zone_to_distance = df.groupby("Zone")["Distance_lisiere"].agg(lambda x: x.mode()[0]).to_dict()
selected_zones = df["Zone"].unique()


def compute_partial_corr(df_sub):

    # Standardisation
    scaler = StandardScaler()
    X_scaled_sub = scaler.fit_transform(df_sub[features_cols])

    # Calcul des distances
    distance_matrix = squareform(pdist(X_scaled_sub, metric='euclidean'))

    lisiere_matrix = np.abs(df_sub["Distance_lisiere"].values[:, None] - df_sub["Distance_lisiere"].values[None, :])
    site_matrix = (df_sub["Zone"].values[:, None] != df_sub["Zone"].values[None, :]).astype(int)

    triu_idx = np.triu_indices_from(distance_matrix, k=1)

    df_corr = pd.DataFrame({
        'euclidean': distance_matrix[triu_idx],
        'lisiere': lisiere_matrix[triu_idx],
        'site': site_matrix[triu_idx]
    })

    pcorr = pg.partial_corr(data=df_corr, x='euclidean', y='lisiere', covar='site', method='spearman')
    return pcorr['r'].values[0],pcorr["CI95%"].values[0]


# Listes de résultats
results_global, results_jour, results_nuit = [], [], []


# Boucle 
for seed in tqdm(range(100), desc="Iterations"):
    np.random.seed(seed)
    random.seed(seed)

    results = []

    for zone in selected_zones:
        df_zone = df[df["Zone"] == zone]

        valid_days = []
        for date, group in df_zone.groupby("Date"):
            if set(group["Intervalle"].dropna().unique()) == set(df["Intervalle"].cat.categories):
                valid_days.append(date)

        if len(valid_days) < 4:
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

    df_result = pd.DataFrame(results)
    if df_result.empty:
        continue

    df_result = df_result.sort_values(by=["Distance_lisiere", "Zone", "Date", "Intervalle"]).reset_index(drop=True)

    df_sub = df_result
    df_sub_jour = df_sub[df_sub["Intervalle"].isin(["6-9h", "9-12h", "12-15h", "15-18h"])]
    df_sub_nuit = df_sub[df_sub["Intervalle"].isin(["21-00h", "0-3h", "3-6h", "18-21h"])]

    try:
        results_global.append(compute_partial_corr(df_sub))
        results_jour.append(compute_partial_corr(df_sub_jour))
        results_nuit.append(compute_partial_corr(df_sub_nuit))
    except Exception as e:
        continue  

def format_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

if results_global:
    r_global = [r[0] for r in results_global]
    ci_global = np.array([r[1] for r in results_global])
    mean_r_global = np.mean(r_global)
    mean_ci_global = np.mean(ci_global, axis=0)

    r_jour = [r[0] for r in results_jour]
    ci_jour = np.array([r[1] for r in results_jour])
    mean_r_jour = np.mean(r_jour)
    mean_ci_jour = np.mean(ci_jour, axis=0)

    r_nuit = [r[0] for r in results_nuit]
    ci_nuit = np.array([r[1] for r in results_nuit])
    mean_r_nuit = np.mean(r_nuit)
    mean_ci_nuit = np.mean(ci_nuit, axis=0)

    print("\n==== MOYENNES DES CORRÉLATIONS PARTIELLES (100 itérations) ====")
    print(f"Global : {mean_r_global:.4f} avec IC95% {format_ci(mean_ci_global)}")
    print(f"Jour   : {mean_r_jour:.4f} avec IC95% {format_ci(mean_ci_jour)}")
    print(f"Nuit   : {mean_r_nuit:.4f} avec IC95% {format_ci(mean_ci_nuit)}")
else:
    print("Pas de résultats valides à afficher.")


