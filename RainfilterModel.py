import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




#Pre Processing

df_indices = pd.read_csv('data',sep=',')


# Target
y = df_indices["rainy (1 = rainy, 0 = not rainy)"]

# Features
excluded_columns=["Fichier", "rainy (1 = rainy, 0 = not rainy)","Zone","Distance_lisiere","Heure","Date","MEANt","jour/nuit","Identifiant"]
X = df_indices.drop(columns=excluded_columns)


# Lists

train_balanced_acc_svc_list = []
test_balanced_acc_svc_list = []

# Number of iterations 
num_seed = 1
random_seeds = np.random.randint(0, 1000, size=num_seed)


for seed in random_seeds:
    
    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    
    #Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Cross-Val
    n_cv = 5
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    
    # GridSearch
    tuned_parameters = {'gamma': np.logspace(-5, 1, num=10), 'C': np.logspace(-1, 4, num=10)}

    
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), tuned_parameters, n_jobs=-1, cv=cv, pre_dispatch=6,
                       scoring="balanced_accuracy", verbose=False)
    
    # PCA 
    n_components = 0.99
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    
    # Explained variance
    # print('Explained variance: ' + str(np.sum(pca.explained_variance_ratio_)))
    
    # Classifier training
    clf.fit(X_train_pca, y_train)

    # Prediction
    y_train_pred_svc=clf.predict(X_train_pca)

    # Balanced Accuracy
    train_balanced_acc_svc = balanced_accuracy_score(y_train, y_train_pred_svc)
   
   
    print('\n--- Training Results ---')
    print(f'SVC - Training Balanced Accuracy: {train_balanced_acc_svc:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_train, y_train_pred_svc))
   
    # Storage
    train_balanced_acc_svc_list.append(train_balanced_acc_svc)

  
     # Test 
     
    #Scale
    X_test_scaled = scaler.transform(X_test)
    
    
    #  PCA
    X_test_pca = pca.transform(X_test_scaled)
    y_test_pred_svc = clf.predict(X_test_pca)
    
    # Balanced Accuracy
    test_balanced_acc_svc = balanced_accuracy_score(y_test, y_test_pred_svc)
    test_balanced_acc_svc_list.append(test_balanced_acc_svc)
  
# PRINT
 
print("\n=== Final Results ===")
print(f'Best SVC parameters: {clf.best_params_}')
# Training results
train_mean_svc, train_std_svc = np.mean(train_balanced_acc_svc_list), np.std(train_balanced_acc_svc_list)
print(f"SVC - Mean Training Balanced Accuracy: {train_mean_svc:.4f} ± {train_std_svc:.4f}")

# Test results
test_mean_svc, test_std_svc = np.mean(test_balanced_acc_svc_list), np.std(test_balanced_acc_svc_list)

print(f"SVC - Mean Test Balanced Accuracy: {test_mean_svc:.4f} ± {test_std_svc:.4f}")

print(' Last Confusion Matrix Test')
print(confusion_matrix(y_test,y_test_pred_svc))


#%% APPLICATION ON THE GLOBAL DATASET


# Load global dataset
df_new = pd.read_csv('globaldataset.csv', sep=',')

X_new = df_new.drop(columns=excluded_columns)


# Scale + PCA

X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)


# Prediction 
y_new_pred_svc = clf.predict(X_new_pca)


# Add to dataframe
df_new["rainy (1 = rainy, 0 = not rainy)"] = y_new_pred_svc


# Save
df_new.to_csv('.csv', sep=';', index=False)
