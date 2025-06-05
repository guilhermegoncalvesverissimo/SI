import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer

import warnings
import random
from utilsAA import *

from sklearn import tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# 3.2 Escolha dos parâmetros e pré-processamentos

# Normalização dos dados (usada para KNN e, opcionalmente, para NB)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Decision Tree: procurar o melhor max_depth
from sklearn.model_selection import GridSearchCV

dt_params = {'max_depth': range(1, 11)}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_grid.fit(X, y)
melhor_max_depth = dt_grid.best_params_['max_depth']

# KNN: procurar o melhor n_neighbors
knn_params = {'n_neighbors': range(1, 21)}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_scaled, y)
melhor_k = knn_grid.best_params_['n_neighbors']

# Naive Bayes: normalmente não tem parâmetros para afinar, mas pode-se testar com e sem normalização
# Aqui, apenas guardamos as versões dos dados para testar depois

# Guardar os melhores parâmetros para usar na criação dos modelos finais
print("Melhor max_depth DT:", melhor_max_depth)
print("Melhor k KNN:", melhor_k)


# 3.3 Estimativas de qualidade

from sklearn.model_selection import StratifiedKFold

def calcular_metricas_cv(modelo, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, mccs = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average='macro'))
        mccs.append(matthews_corrcoef(y[test_idx], y_pred))
    return np.array(accs), np.array(f1s), np.array(mccs)

def estatisticas(m):
    q1 = np.percentile(m, 25)
    q2 = np.percentile(m, 50)
    q3 = np.percentile(m, 75)
    adj_sup = np.min([np.max(m[m <= q3 + 1.5*(q3-q1)]), np.max(m)])
    adj_inf = np.max([np.min(m[m >= q1 - 1.5*(q3-q1)]), np.min(m)])
    return [q1, q2, q3, adj_sup, adj_inf, np.max(m), np.min(m)]

# DT
dt_model = DecisionTreeClassifier(max_depth=melhor_max_depth, random_state=42)
acc_dt, f1_dt, mcc_dt = calcular_metricas_cv(dt_model, X.values, y.values)
dt_acc_stats = estatisticas(acc_dt)
dt_f1_stats = estatisticas(f1_dt)
dt_mcc_stats = estatisticas(mcc_dt)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=melhor_k)
acc_knn, f1_knn, mcc_knn = calcular_metricas_cv(knn_model, X_scaled, y.values)
knn_acc_stats = estatisticas(acc_knn)
knn_f1_stats = estatisticas(f1_knn)
knn_mcc_stats = estatisticas(mcc_knn)

# NB (usando normalização, mas podes testar sem)
nb_model = GaussianNB()
acc_nb, f1_nb, mcc_nb = calcular_metricas_cv(nb_model, X_scaled, y.values)
nb_acc_stats = estatisticas(acc_nb)
nb_f1_stats = estatisticas(f1_nb)
nb_mcc_stats = estatisticas(mcc_nb)

# Agora tens as variáveis:
# dt_acc_stats, dt_f1_stats, dt_mcc_stats
# knn_acc_stats, knn_f1_stats, knn_mcc_stats
# nb_acc_stats, nb_f1_stats, nb_mcc_stats
# Cada uma é uma lista: [Q1, Q2, Q3, adj_sup, adj_inf, max, min]

# 3.4 Criação dos modelos finais

# Decision Tree com melhor max_depth encontrado
modelo_final_DT = DecisionTreeClassifier(max_depth=melhor_max_depth, random_state=42)
modelo_final_DT.fit(X, y)

# KNN com melhor k encontrado (usa dados normalizados)
modelo_final_KNN = KNeighborsClassifier(n_neighbors=melhor_k)
modelo_final_KNN.fit(X_scaled, y)

# Naive Bayes (usa dados normalizados)
modelo_final_NB = GaussianNB()
modelo_final_NB.fit(X_scaled, y)

# 3.5 RESULTADOS


# É ABSOLUTAMENTE NECESSÁRIO indicar o número do grupo:
GRUPO = 22  # <--- coloca aqui o teu número de grupo

# Podem alterar o nome do ficheiro se quiserem testar com outros ficheiros
nome_ficheiro_dados_novos = "DadosNovos.csv"
# Daqui para a frente, não se usa o nome DadosNovos.csv mas apenas a variável nome_ficheiro_dados_novos

# NÃO ALTERAR estas linhas:
# Aqui definimos o nome do ficheiro de resultados que devem gravar.
FICHEIRO = nome_ficheiro_dados_novos.split(".")[0]
nome_ficheiro_resultados = "Resultados_"+FICHEIRO+"_SInt_24_25_grupo"+str(GRUPO)+".txt"
# Daqui para a frente, usa-se a variável nome_ficheiro_resultados para identificar o ficheiro de resultados

print(GRUPO)
print(nome_ficheiro_dados_novos)
print(nome_ficheiro_resultados)

# Se desenvolverem algum código extra aqui,
# também o devem copiar para o ficheiro Python Aprendizagem_SInt_24_25_grupoXX.py,
# onde substituem o XX pelo número do vosso grupo.

# --- AQUI COMEÇA O CÓDIGO PARA GERAR O FICHEIRO DE RESULTADOS ---

dados_novos = pd.read_csv(nome_ficheiro_dados_novos, header=None)
dados_novos_scaled = scaler.transform(dados_novos)

prev_DT = modelo_final_DT.predict(dados_novos)
prev_KNN = modelo_final_KNN.predict(dados_novos_scaled)
prev_NB = modelo_final_NB.predict(dados_novos_scaled)

with open(nome_ficheiro_resultados, "w") as f:
    f.write(f"{GRUPO}\n")
    f.writelines([f"{v}\n" for v in dt_acc_stats])
    f.writelines([f"{v}\n" for v in dt_f1_stats])
    f.writelines([f"{v}\n" for v in dt_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_DT])
    f.writelines([f"{v}\n" for v in knn_acc_stats])
    f.writelines([f"{v}\n" for v in knn_f1_stats])
    f.writelines([f"{v}\n" for v in knn_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_KNN])
    f.writelines([f"{v}\n" for v in nb_acc_stats])
    f.writelines([f"{v}\n" for v in nb_f1_stats])
    f.writelines([f"{v}\n" for v in nb_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_NB])

#-----------------------#-------------------------#
dados_novos = pd.read_csv(nome_ficheiro_dados_novos, header=None)
dados_novos_scaled = scaler.transform(dados_novos)

prev_DT = modelo_final_DT.predict(dados_novos)
prev_KNN = modelo_final_KNN.predict(dados_novos_scaled)
prev_NB = modelo_final_NB.predict(dados_novos_scaled)

with open(nome_ficheiro_resultados, "w") as f:
    f.write(f"{GRUPO}\n")
    f.writelines([f"{v}\n" for v in dt_acc_stats])
    f.writelines([f"{v}\n" for v in dt_f1_stats])
    f.writelines([f"{v}\n" for v in dt_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_DT])
    f.writelines([f"{v}\n" for v in knn_acc_stats])
    f.writelines([f"{v}\n" for v in knn_f1_stats])
    f.writelines([f"{v}\n" for v in knn_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_KNN])
    f.writelines([f"{v}\n" for v in nb_acc_stats])
    f.writelines([f"{v}\n" for v in nb_f1_stats])
    f.writelines([f"{v}\n" for v in nb_mcc_stats])
    f.writelines([f"{int(v)}\n" for v in prev_NB])