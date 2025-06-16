import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def bubble_model_analysis(df, target, features_cols, nb_trials=10000, nb_bubbles=15, model_type='ridge'):
    nb_indices = len(features_cols)
    results = {
        'r2_scores': np.zeros(nb_trials),
        'bubble_masks': np.zeros((nb_trials, nb_indices))
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features_cols])
    y = target.values

    for i in tqdm(range(nb_trials), desc="Bubble Model Analysis"):
        bubble_indices = np.random.choice(nb_indices, nb_bubbles, replace=False)
        bubble_vec = np.zeros(nb_indices)
        bubble_vec[bubble_indices] = 1.0

        results['bubble_masks'][i, :] = bubble_vec

        X_masked = X * bubble_vec

        X_train, X_test, y_train, y_test = train_test_split(X_masked, y, train_size=1000)

        pca = PCA(n_components=0.99)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'svr':
            # Adjust with best params for the model deducted before
            model = SVR(C=215.0, gamma=0.1134)

        try:
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            r2 = r2_score(y_test, y_pred)
        except Exception as e:
            r2 = np.nan

        results['r2_scores'][i] = r2

    return results


def correlate_bubbles_with_model_perf(results):
    correlations = []
    for i in range(results['bubble_masks'].shape[1]):
        valid_mask = ~np.isnan(results['r2_scores'])
        corr, _ = pearsonr(results['bubble_masks'][valid_mask, i], results['r2_scores'][valid_mask])
        correlations.append(corr)
    return np.array(correlations)




features_cols = X.columns.tolist()

# SVR
results_svr = bubble_model_analysis(df_indices, y, features_cols, nb_trials=10000, nb_bubbles=15, model_type='svr')
correlations_svr = correlate_bubbles_with_model_perf(results_svr)

# Ridge
results_ridge = bubble_model_analysis(df_indices, y, features_cols, nb_trials=10000, nb_bubbles=15, model_type='ridge')
correlations_ridge = correlate_bubbles_with_model_perf(results_ridge)

df_corr = pd.DataFrame({
    'Feature': features_cols,
    'SVR': correlations_svr,
    'Ridge': correlations_ridge
})

df_corr_melted = df_corr.melt(id_vars='Feature', var_name='Model', value_name='Correlation')
df_corr_melted = df_corr_melted.sort_values(by='Feature')

plt.figure(figsize=(14, 6))
sns.barplot(data=df_corr_melted, x='Feature', y='Correlation', hue='Model')
plt.title("Importance des indices acoustiques selon les modèles (impact sur R²)", fontsize=16)
plt.xlabel("Indice acoustique", fontsize=14)
plt.ylabel("Importance sur R²", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(x=features_cols, y=correlations_svr, color='blue', alpha=0.7)
plt.title("Importance des indices acoustiques", fontsize=16)
plt.xlabel("Indice acoustique", fontsize=14)
plt.ylabel("Corrélation", fontsize=14)
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  


plt.show()

# Negative indices
negative_indices = [f'"{feature}"' for feature, corr in zip(features_cols, correlations_svr) if corr < 0]

