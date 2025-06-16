import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# load df
df_indices = pd.read_csv('data.csv', sep=',')

# Nettoyage de la colonne Distance_lisiere
df_indices['Distance_lisiere'] = df_indices['Distance_lisiere'].str.extract(r'(\d+)')
df_indices['Distance_lisiere'] = pd.to_numeric(df_indices['Distance_lisiere'])

# Filtres
df_indices = df_indices[df_indices["rainy (1 = rainy, 0 = not rainy)"] == 0]

#JOUR
# df_indices = df_indices[(df_indices["Heure"] >= 50000) & (df_indices["Heure"] < 180000)]
#NUIT
# df_indices = df_indices[(df_indices["Heure"] >= 180000) | (df_indices["Heure"] < 50000)]



# Keep only features columns
X = df_indices.drop(columns=["Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone", "Heure", "Date", "jour/nuit", "Identifiant"])

# Liste des indices
all_indices = [col for col in X.columns if col != 'Distance_lisiere']
n_indices = len(all_indices)
plots_per_figure = 6

# Liste pour stocker les indices significatifs avec leur r
significant_indices = []

sns.set(style="whitegrid")

for start in range(0, n_indices, plots_per_figure):
    end = min(start + plots_per_figure, n_indices)
    subset_indices = all_indices[start:end]

    n_cols = 2
    n_rows = math.ceil(len(subset_indices) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, index in enumerate(subset_indices):
        ax = axes[i]

        sns.boxplot(
            x='Distance_lisiere',
            y=index,
            data=df_indices,
            ax=ax,
            color='skyblue',
            showfliers=False
        )

        r, p_value = spearmanr(df_indices['Distance_lisiere'], df_indices[index], nan_policy='omit')

        # Conditions 
        if p_value < 0.05 and abs(r) >= 0.3:
            significant_indices.append((index, r))
            title_color = "green"
        else:
            title_color = "red"

        ax.set_title(f'Indice: {index}\n r = {r:.2f}, p = {p_value:.3f}', color=title_color)
        ax.set_xlabel('Distance à la lisière (m)')
        ax.set_ylabel('Valeur')

    plt.tight_layout()
    plt.show()

# Affichage

print("Indices avec une corrélation significative :")
for idx, r_val in significant_indices:
    print(f"- {idx} : r = {r_val:.2f}")
    
    
    
# Filtrer uniquement les indices significatifs
significant_names = [name for name, _ in significant_indices]
n_sig = len(significant_names)
plots_per_figure = 6

for start in range(0, n_sig, plots_per_figure):
    end = min(start + plots_per_figure, n_sig)
    subset_indices = significant_names[start:end]

    n_cols = 2
    n_rows = math.ceil(len(subset_indices) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, index in enumerate(subset_indices):
        ax = axes[i]

        sns.boxplot(
            x='Distance_lisiere',
            y=index,
            data=df_indices,
            ax=ax,
            color='skyblue',
            showfliers=False
        )

        r, p_value = spearmanr(df_indices['Distance_lisiere'], df_indices[index], nan_policy='omit')
        ax.set_title(f'Indice: {index}\n r = {r:.2f}, p = {p_value:.3f}', color='green')
        ax.set_xlabel('Distance à la lisière (m)')
        ax.set_ylabel('Valeur')

    for j in range(len(subset_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


