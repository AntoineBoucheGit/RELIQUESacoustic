import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import pingouin as pg
import random
from tqdm import tqdm
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
#Global
df_filtered=df_result

# JOUR
# intervals_to_keep = ["6-9h","9-12h","12-15h","15-18h"]  
# df_filtered = df_result[df_result["Intervalle"].isin(intervals_to_keep)].copy()

# NUIT
# intervals_to_keep = ["18-21h","21-00h","0-3h","3-6h"]  
# df_filtered = df_result[df_result["Intervalle"].isin(intervals_to_keep)].copy()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_filtered[features_cols])
df_scaled = df_filtered.copy()
df_scaled[features_cols] = scaled_features


distances_lisiere = sorted(df_scaled["Distance_lisiere"].unique())
print(f"Distances à la lisière dans le jeu de données: {distances_lisiere}")


n_samples = 25
correlations = []


for distance_1 in distances_lisiere:
    for distance_2 in distances_lisiere:
        if distance_1 == distance_2:
            continue
            
        print(f"Analyse des distances {distance_1}m et {distance_2}m...")
        sample_correlations = []
        

        df1_full = df_scaled[df_scaled["Distance_lisiere"] == distance_1]
        df2_full = df_scaled[df_scaled["Distance_lisiere"] == distance_2]
        

        min_size = min(len(df1_full), len(df2_full))
        if min_size < 5:
            print(f"  Pas assez de données pour les distances {distance_1}m et {distance_2}m")
            correlations.append((distance_1, distance_2, np.nan))
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
        correlations.append((distance_1, distance_2, mean_corr, std_corr))


df_correlations = pd.DataFrame(correlations, columns=["Distance_1", "Distance_2", "Correlation", "Std_Correlation"])


df_pivot = df_correlations.pivot(index="Distance_1", columns="Distance_2", values="Correlation")

plt.figure(figsize=(12, 14))
sns.heatmap(df_pivot, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.xlabel("Distance à la lisière (m)", fontsize=12)
plt.ylabel("Distance à la lisière (m)", fontsize=12)
plt.tight_layout()
plt.show()




#%% SCHEMA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.cm as cm
from itertools import combinations




distances_sorted = sorted(distances_lisiere)
x_positions = {d: i * 1.5 for i, d in enumerate(distances_sorted)}

# Normalisation
corr_values = df_correlations['Correlation'].dropna()
norm_color = plt.Normalize(vmin=corr_values.min(), vmax=corr_values.max())
norm_width = plt.Normalize(vmin=0, vmax=abs(corr_values).max())
cmap = cm.get_cmap('viridis')


fig, ax = plt.subplots(figsize=(13, 6))

draw_up = True
seen_pairs = set()

for _, row in df_correlations.iterrows():
    d1, d2, corr = row['Distance_1'], row['Distance_2'], row['Correlation']
    if np.isnan(corr) or d1 == d2:
        continue

    pair = tuple(sorted((d1, d2)))
    if pair in seen_pairs:
        continue
    seen_pairs.add(pair)

    x1, x2 = x_positions[pair[0]], x_positions[pair[1]]
    mid_x = (x1 + x2) / 2
    arc_height = 0.2 * abs(x2 - x1)
    mid_y = arc_height if draw_up else -arc_height
    draw_up = not draw_up  

    color = cmap(norm_color(corr))
    width = 0.5 + 3 * norm_width(abs(corr))  

    path = Path([(x1, 0), (mid_x, mid_y), (x2, 0)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    patch = PathPatch(path, edgecolor=color, lw=width, facecolor='none', alpha=0.9)
    ax.add_patch(patch)


for d, x in x_positions.items():
    ax.plot(x, 0, 'o', color='black', markersize=14)
    ax.text(x, -0.2, f"{d} m", ha='center', va='top', fontsize=10)

sm = cm.ScalarMappable(norm=norm_color, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.1)
cbar.set_label('Corrélation', fontsize=30)


ax.set_xlim(min(x_positions.values()) - 1, max(x_positions.values()) + 1)
ax.set_ylim(-max(x_positions.values()) * 0.3, max(x_positions.values()) * 0.3)
ax.axis('off')

plt.tight_layout()
plt.show()
