import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Données
df_indices = pd.read_csv('/Users/superman/Desktop/Stage/IndicesReliques/ReliquesIndicesTot.csv', sep=',')

# SI Distance lisiere pas sous format numérique
df_indices['Distance_lisiere'] = df_indices['Distance_lisiere'].str.extract(r'(\d+)')

df_indices['Distance_lisiere'] = pd.to_numeric(df_indices['Distance_lisiere'])

#SI on filtre
df_indices = df_indices[df_indices["rainy (1 = rainy, 0 = not rainy)"] == 0]


# Target 
y = df_indices["Distance_lisiere"]

# Features
X = df_indices.drop(columns=["Fichier", "rainy (1 = rainy, 0 = not rainy)", "Zone", "Distance_lisiere", "Heure", "Date","jour/nuit","MEANt"])


# Nombres d'itérations 
num_splits = 5 
random_seeds = np.random.randint(0, 1000, size=num_splits)

# Listes Résultats
ridge_train_rmse_list = []
ridge_train_r2_list = []
svr_C_list = []
svr_gamma_list = []
svr_train_rmse_list = []
svr_train_r2_list = []
ridge_test_rmse_list = []
ridge_test_r2_list = []
svr_test_rmse_list = []
svr_test_r2_list = []



for seed in random_seeds:
    
    # Split Train/Test
    # Adapt train size to your dataset size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000)
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    
    # Cross-Val
    n_cv = 5
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=seed)
    
    # GridSearch
    ridge_parameters = {'alpha': np.logspace(-3, 3, num=10)}
    svr_parameters = {
        'C': np.logspace(-1, 4, 10),
        'gamma': np.logspace(-3, 1, 10),
        'epsilon': [0.01, 0.1, 0.2]
    }

  # Ridge model (Linear regression)
    ridge = GridSearchCV(Ridge(), ridge_parameters, n_jobs=-1, cv=cv, pre_dispatch=6,
                        scoring='neg_mean_squared_error', verbose=False)

  # SVR model (Support Vector Regression ) (non-linear regression)
    svr = GridSearchCV(SVR(kernel="rbf"), svr_parameters, n_jobs=-1, cv=cv, pre_dispatch=6,
                      scoring='neg_mean_squared_error', verbose=False)
    
    # PCA
    n_components = 0.99
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    print(f'Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}')
       
   
    
    # Train with PCA data
    ridge.fit(X_train_pca, y_train)
    svr.fit(X_train_pca, y_train)
    
    # TRAINING EVALUATION
    print("\n--- Training Results ---")
    y_train_pred_ridge = ridge.predict(X_train_pca)
    y_train_pred_svr = svr.predict(X_train_pca)
    
    # Train R2
    train_score_svr = r2_score(y_train, y_train_pred_svr)
    train_score_ridge = r2_score(y_train,y_train_pred_ridge)
    
    # Train RMSE
    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)) 
    svr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_svr))


    print(f'Ridge - Training RMSE: {ridge_train_rmse:.4f}, R²: {train_score_ridge:.4f}')
    print(f'SVR - Training RMSE: {svr_train_rmse:.4f}, R²: {train_score_svr:.4f}')
    
    ridge_train_rmse_list.append(ridge_train_rmse)
    ridge_train_r2_list.append(train_score_ridge)
    svr_train_rmse_list.append(svr_train_rmse)
    svr_train_r2_list.append(train_score_svr)
    svr_C_list.append(svr.best_params_['C'])
    svr_gamma_list.append(svr.best_params_['gamma'])
    
    print(f'Best SVR parameters: {svr.best_params_}')
    
    

    # TEST EVALUATION
    
    # Scale et PCA
    
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    y_test_pred_ridge = ridge.predict(X_test_pca)
    y_test_pred_svr = svr.predict(X_test_pca)
    
    print("\n--- Test Results ---")

    
    # Test R2
    svr_test_r2 = r2_score(y_test, y_test_pred_svr)
    ridge_test_r2 = r2_score(y_test,y_test_pred_ridge)
    
    # test RMSE
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)) 
    svr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_svr))


    print(f'Ridge - Test RMSE: {ridge_test_rmse:.4f}, R²: {ridge_test_r2:.4f}')
    print(f'SVR - Test RMSE: {svr_test_rmse:.4f}, R²: {svr_test_r2:.4f}')
    print("\n-------------------------------------------------")
    
    ridge_test_rmse_list.append(ridge_test_rmse)
    ridge_test_r2_list.append(ridge_test_r2)
    svr_test_rmse_list.append(svr_test_rmse)
    svr_test_r2_list.append(svr_test_r2)
 

    
    
# Graphics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  
#  Ridge
sns.regplot(x=y_test, y=y_test_pred_ridge, ax=axes[0], scatter_kws={'s':10}, line_kws={"color": "red"})
 
axes[0].plot([0, 300], [0, 300], linestyle="--", color="black", label="x = y")
axes[0].set_title("Ridge Regression: Réel vs Prédiction")
axes[0].set_xlabel("Distance réelle")
axes[0].set_ylabel("Distance prédite")
axes[0].text(0.05, 0.9, f"$rmse = {ridge_test_rmse:.2f}$\n$R^2 = {ridge_test_r2:.2f}$",
             transform=axes[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

#  SVR
sns.regplot(x=y_test, y=y_test_pred_svr, ax=axes[1], scatter_kws={'s':10}, line_kws={"color": "red"})
axes[1].plot([0, 300], [0, 300], linestyle="--", color="black", label="x = y")
axes[1].set_title("SVR: Réel vs Prédiction")
axes[1].set_xlabel("Distance réelle")
axes[1].set_ylabel("Distance prédite")
axes[1].text(0.05, 0.9, f"$rmse = {svr_test_rmse:.2f}$\n$R^2 = {svr_test_r2:.2f}$",
             transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

        



# Ridge
print("\n--- Ridge ---")
ridge_train_mean_rmse, ridge_train_std_rmse = np.mean(ridge_train_rmse_list), np.std(ridge_train_rmse_list)
ridge_train_mean_r2, ridge_train_std_r2 = np.mean(ridge_train_r2_list), np.std(ridge_train_r2_list)
ridge_test_mean_rmse, ridge_test_std_rmse = np.mean(ridge_test_rmse_list), np.std(ridge_test_rmse_list)
ridge_test_mean_r2, ridge_test_std_r2 = np.mean(ridge_test_r2_list), np.std(ridge_test_r2_list)

print(f"Ridge - Mean Training RMSE: {ridge_train_mean_rmse:.4f} ± {ridge_train_std_rmse:.4f}")
print(f"Ridge - Mean Training R²: {ridge_train_mean_r2:.4f} ± {ridge_train_std_r2:.4f}")
print(f"Ridge - Mean Test RMSE: {ridge_test_mean_rmse:.4f} ± {ridge_test_std_rmse:.4f}")
print(f"Ridge - Mean Test R²: {ridge_test_mean_r2:.4f} ± {ridge_test_std_r2:.4f}")
# SVR
print("\n--- SVR ---")
svr_train_mean_rmse, svr_train_std_rmse = np.mean(svr_train_rmse_list), np.std(svr_train_rmse_list)
svr_train_mean_r2, svr_train_std_r2 = np.mean(svr_train_r2_list), np.std(svr_train_r2_list)
svr_test_mean_rmse, svr_test_std_rmse = np.mean(svr_test_rmse_list), np.std(svr_test_rmse_list)
svr_test_mean_r2, svr_test_std_r2 = np.mean(svr_test_r2_list), np.std(svr_test_r2_list)
svr_mean_C,svr_std_C=np.mean(svr_C_list), np.std(svr_C_list)
svr_mean_gamma,svr_std_gamma=np.mean(svr_gamma_list), np.std(svr_gamma_list)


print(f"SVR - Mean Training RMSE: {svr_train_mean_rmse:.4f} ± {svr_train_std_rmse:.4f}")
print(f"SVR - Mean Training R²: {svr_train_mean_r2:.4f} ± {svr_train_std_r2:.4f}")
print(f"SVR - Mean Test RMSE: {svr_test_mean_rmse:.4f} ± {svr_test_std_rmse:.4f}")
print(f"SVR - Mean Test R²: {svr_test_mean_r2:.4f} ± {svr_test_std_r2:.4f}")
print(f"SVR - Mean C Param: {svr_mean_C:.4f} ± {svr_std_C:.4f}")
print(f"SVR - Mean gamma Param: {svr_mean_gamma:.4f} ± {svr_std_gamma:.4f}")
