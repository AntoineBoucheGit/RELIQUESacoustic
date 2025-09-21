import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import pingouin as pg
from tqdm import tqdm

# df avec indices acoustiques pour chaque site ,6jours 8 intervalles horaires
df = df_result


# Calcul des matrices

def calculate_dissimilarity_matrices(df, features_cols):

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_cols])
    df_scaled = df.copy()
    df_scaled[features_cols] = X_scaled

    # MATRICE DE DISTANCE acoustic
    dist_mat = squareform(pdist(df_scaled[features_cols], metric='acoustic'))

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
    dist_mat_masked = squareform(pdist(masked_data, metric='acoustic'))

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
