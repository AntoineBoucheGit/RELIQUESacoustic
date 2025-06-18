import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import random
from scipy.stats import pearsonr


np.random.seed(44)
random.seed(44)


df = pd.read_csv('/Users/superman/Desktop/Stage/IndicesReliques/ReliquesIndicesTotAENettoyé.csv', sep=',')
df['Distance_lisiere'] = pd.to_numeric(df['Distance_lisiere'].astype(str).str.extract(r'(\d+)')[0])
df = df[df["rainy (1 = rainy, 0 = not rainy)"] == 0]
df["Heure_num"] = df["Heure"].astype(str).str.zfill(6).str[:2].astype(int)


df["Intervalle"] = pd.cut(df["Heure_num"],
                          bins=[0, 3, 6, 9, 12, 15, 18, 21,24],
                          labels=["0-3h", "3-6h", "6-9h", "9-12h", "12-15h", "15-18h", "18-21h", "21-00h"],
                          right=False, include_lowest=True)


features_cols = df.columns.difference([
    "Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone",
    "Distance_lisiere", "Heure", "Heure_num", "Date",
    "jour/nuit", "MEANt", "Intervalle", "Identifiant"
])

zone_to_distance = df.groupby("Zone")["Distance_lisiere"].agg(lambda x: x.mode()[0]).to_dict()


n_repetitions = 10 
all_results = []

for rep in range(n_repetitions):
    seed_value = 44 + rep
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    results = []
    selected_zones = df["Zone"].unique()  

    for zone in selected_zones:
        df_zone = df[df["Zone"] == zone]

        valid_days = []
        for date, group in df_zone.groupby("Date"):
            if set(group["Intervalle"].dropna().unique()) == set(df["Intervalle"].cat.categories):
                valid_days.append(date)

        if len(valid_days) < 4:
            print(f"Zone {zone} ignorée (pas assez de jours valides : {len(valid_days)})")
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


    df_rep = pd.DataFrame(results)
    df_rep = df_rep.sort_values(by=["Distance_lisiere", "Zone", "Date", "Intervalle"]).reset_index(drop=True)
    all_results.append(df_rep)


df_result = pd.concat(all_results, ignore_index=True)

#INTERVALLES CHOISIS
df_filtered=df_result
# intervals_to_keep = ["12-15h", "15-18h"]  
# df_filtered = df_result[df_result["Intervalle"].isin(intervals_to_keep)].copy()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_filtered[features_cols])
df_scaled = df_filtered.copy()
df_scaled[features_cols] = scaled_features


zones = sorted(df_scaled["Zone"].unique())
print(f"Zones dans le jeu de données: {zones}")

n_samples = 25
correlations = []

for zone_1 in zones:
    for zone_2 in zones:
        if zone_1 == zone_2:
            continue
            
        print(f"Analyse des zones {zone_1} et {zone_2}...")
        sample_correlations = []
        

        df1_full = df_scaled[df_scaled["Zone"] == zone_1]
        df2_full = df_scaled[df_scaled["Zone"] == zone_2]
        
        min_size = min(len(df1_full), len(df2_full))
        if min_size < 5:
            print(f"  Pas assez de données pour les zones {zone_1} et {zone_2}")
            correlations.append((zone_1, zone_2, np.nan))
            continue
        
        for i in range(n_samples):
            sample_seed = 100 + i
            
            df1_sampled = df1_full.sample(n=min_size, random_state=sample_seed)
            df2_sampled = df2_full.sample(n=min_size, random_state=sample_seed)
            

            dist1 = squareform(pdist(df1_sampled[features_cols], metric='euclidean'))
            dist2 = squareform(pdist(df2_sampled[features_cols], metric='euclidean'))
            

            triu_indices = np.triu_indices_from(dist1, k=1)
            

            corr, _ = pearsonr(dist1[triu_indices], dist2[triu_indices])
            sample_correlations.append(corr)
        

        mean_corr = np.mean(sample_correlations)
        std_corr = np.std(sample_correlations)
        
        print(f"  Corrélation moyenne: {mean_corr:.4f}, écart-type: {std_corr:.4f}")
        correlations.append((zone_1, zone_2, mean_corr, std_corr))


df_correlations = pd.DataFrame(correlations, columns=["Zone_1", "Zone_2", "Correlation", "Std_Correlation"])
df_pivot = df_correlations.pivot(index="Zone_1", columns="Zone_2", values="Correlation")

zone_distances = {zone: zone_to_distance[zone] for zone in df_pivot.index}
sorted_zones = sorted(zone_distances, key=lambda x: zone_distances[x])

df_pivot_sorted = df_pivot.loc[sorted_zones, sorted_zones]


plt.figure(figsize=(10, 6))

ax = sns.heatmap(
    df_pivot_sorted, 
    cmap="viridis", 
    fmt=".2f", 
    linewidths=0.5, 
    vmax=0.4,
    cbar_kws={'label': 'Corrélation'}
)

ax.set_xlabel("Site", fontsize=20)
ax.set_ylabel("Site", fontsize=20)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Corrélation", fontsize=30)

plt.tight_layout()
plt.show()

