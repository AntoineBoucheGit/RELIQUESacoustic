
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import pingouin as pg
from tqdm import tqdm
import seaborn as sns
import random


np.random.seed(44)
random.seed(44)


df = pd.read_csv('/Users/superman/Desktop/Projet stage/data/ReliquesIndicesTotAENettoyé.csv', sep=',')
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
    "jour/nuit", "MEANt", "Intervalle", "Identifiant"])

# "ACI","ACTTspCount","ACTspFract","ACTspMean","ACTtCount","ACTtFraction","ACTtMean","EAS","ECU","ECV","EVNspCount","EVNspFract","SNRf"

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

# df avec indices acoustiques pour chaque site ,6jours 8 intervalles horaires
df = df_result



def calculate_dissimilarity_matrices(df, features_cols):

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_cols])
    df_scaled = df.copy()
    df_scaled[features_cols] = X_scaled

    # MATRICE DE DISTANCE acoustic
    dist_mat = squareform(pdist(df_scaled[features_cols], metric='euclidean'))

    # MATRICE DE DIFFÉRENCE DE DISTANCE À LA LISIÈRE
    lis_mat = np.zeros((len(df), len(df)))

    for i in range(len(df)):
        for j in range(len(df)):
            lis_mat[i, j] = abs(df["Distance_lisiere"].iloc[i] - df["Distance_lisiere"].iloc[j]) if i != j else 0


    # MATRICE SITE
    site_mat = np.zeros((len(df), len(df)))

    for i in range(len(df)):
        for j in range(len(df)):
            site_mat[i, j] = 1 if df["Zone"].iloc[i] != df["Zone"].iloc[j] else 0

    return dist_mat, lis_mat, site_mat


# Corrélation Partielle

def corrcompute(masked_data):
    
    # Calcul matrice acoustique avec masque
    dist_mat_masked = squareform(pdist(masked_data, metric='euclidean'))

    # Correlation Partielle
    triu_idx = np.triu_indices_from(dist_mat_masked, k=1)
    df_corr_sub = pd.DataFrame({
        'acoustic': dist_mat_masked[triu_idx],
        'lisiere': lis_mat[triu_idx],
        'site': site_mat[triu_idx]
    })
    
    corr_result = pg.partial_corr(data=df_corr_sub, x='acoustic', y='lisiere', covar='site', method='spearman')
    
    return corr_result['r'].values[0]



def bubble_analysis(df, nb_trials, nb_bubbles, dist_mat, lis_mat, site_mat):
    nb_indices = df[features_cols].shape[1]
    results = {'avg_dissim': np.zeros(nb_trials), 'bubble_masks': np.zeros((nb_trials, nb_indices))}
    
    for i in tqdm(range(nb_trials), desc="Analyse par bulles"):
        # Création du masque de bubles 
        bubble_indices = np.random.choice(nb_indices, nb_bubbles, replace=False) 
        bubble_vec = np.zeros(nb_indices)  # np.zeros ou np.ones??
        bubble_vec[bubble_indices] = np.random.rand(nb_bubbles)  
        
        results['bubble_masks'][i, :] = bubble_vec  

        # masque sur toutes les lignes 
        masked_data = df[features_cols].values * bubble_vec 

        # Calcul corr partielle pour données masquées
        dissimilarity = corrcompute(masked_data)
        results['avg_dissim'][i] = dissimilarity  

    return results


def correlate_bubbles_with_dissim(results):
    
    correlations = []
    
    for i in range(results['bubble_masks'].shape[1]):
        corr, _ = pearsonr(results['bubble_masks'][:, i], results['avg_dissim'])
        correlations.append(corr)
    return np.array(correlations)


# Importance Globale

#JOUR
# df = df_result[df_result["Intervalle"].isin(["6-9h", "9-12h", "12-15h", "15-18h"])]
#NUIT
# df = df_result[df_result["Intervalle"].isin(["0-3h", "3-6h", "18-21h", "21-00h"])]


dist_mat, lis_mat, site_mat = calculate_dissimilarity_matrices(df, features_cols)


scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_cols]), columns=df[features_cols].columns)


results = bubble_analysis(df_scaled, nb_trials=10000, nb_bubbles=15, dist_mat=dist_mat, lis_mat=lis_mat, site_mat=site_mat)
corrs = correlate_bubbles_with_dissim(results)

plt.figure(figsize=(14, 6))
sns.barplot(x=features_cols, y=corrs, color='blue', alpha=0.7)
plt.title("Importance des indices acoustiques", fontsize=16)
plt.xlabel("Indice acoustique", fontsize=14)
plt.ylabel("Corrélation", fontsize=14)
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  


plt.show()

# Negative indices
negative_indices = [f'"{feature}"' for feature, corr in zip(features_cols, corrs) if corr < 0]
