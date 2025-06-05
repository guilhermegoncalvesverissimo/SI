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



data = pd.read_csv('DadosAprendizagem.csv', header=None) 
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import GridSearchCV

dt_params = {'max_depth': range(1, 11)}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_grid.fit(X, y)
melhor_max_depth = dt_grid.best_params_['max_depth']

knn_params = {'n_neighbors': range(1, 21)}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_scaled, y)
melhor_k = knn_grid.best_params_['n_neighbors']

print("Melhor max_depth DT:", melhor_max_depth)
print("Melhor k KNN:", melhor_k)



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
    max_val = np.max(m)
    min_val = np.min(m)
    
    return [min_val, adj_inf, q1, q2, q3, adj_sup, max_val]

dt_model = DecisionTreeClassifier(max_depth=melhor_max_depth, random_state=42)
acc_dt, f1_dt, mcc_dt = calcular_metricas_cv(dt_model, X.values, y.values)
dt_acc_stats = estatisticas(acc_dt)
dt_f1_stats = estatisticas(f1_dt)
dt_mcc_stats = estatisticas(mcc_dt)

knn_model = KNeighborsClassifier(n_neighbors=melhor_k)
acc_knn, f1_knn, mcc_knn = calcular_metricas_cv(knn_model, X_scaled, y.values)
knn_acc_stats = estatisticas(acc_knn)
knn_f1_stats = estatisticas(f1_knn)
knn_mcc_stats = estatisticas(mcc_knn)

nb_model = GaussianNB()
acc_nb, f1_nb, mcc_nb = calcular_metricas_cv(nb_model, X_scaled, y.values)
nb_acc_stats = estatisticas(acc_nb)
nb_f1_stats = estatisticas(f1_nb)
nb_mcc_stats = estatisticas(mcc_nb)

modelo_final_DT = DecisionTreeClassifier(max_depth=melhor_max_depth, random_state=42)
modelo_final_DT.fit(X, y)

modelo_final_KNN = KNeighborsClassifier(n_neighbors=melhor_k)
modelo_final_KNN.fit(X_scaled, y)

modelo_final_NB = GaussianNB()
modelo_final_NB.fit(X_scaled, y)



GRUPO = 22  # <--- coloca aqui o teu número de grupo

nome_ficheiro_dados_novos = "DadosNovos.csv"

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

# Função para escrever as métricas no formato correto
def escreve_metricas(stats, f):
    print("min:", stats[0])
    print("adj_inf:", stats[1])
    print("Q1:", stats[2])
    print("Q2:", stats[3])
    print("Q3:", stats[4])
    print("adj_sup:", stats[5])
    print("max:", stats[6])

    ordem_verificador = [2, 3, 4, 5, 1, 6, 0]
    for i in ordem_verificador:
        f.write(f"{stats[i]}\n")

# Grava os resultados no formato especificado
with open(nome_ficheiro_resultados, "w") as f:
    f.write(f"{GRUPO}\n")
    escreve_metricas(dt_acc_stats, f)
    escreve_metricas(dt_f1_stats, f)
    escreve_metricas(dt_mcc_stats, f)
    f.writelines([f"{int(v)}\n" for v in prev_DT])
    escreve_metricas(knn_acc_stats, f)
    escreve_metricas(knn_f1_stats, f)
    escreve_metricas(knn_mcc_stats, f)
    f.writelines([f"{int(v)}\n" for v in prev_KNN])
    escreve_metricas(nb_acc_stats, f)
    escreve_metricas(nb_f1_stats, f)
    escreve_metricas(nb_mcc_stats, f)
    f.writelines([f"{int(v)}\n" for v in prev_NB])





# NÃO ALTERAR ESTE CÓDIGO
# -----------------------
erros = False
n_medidas_estimativa = 7
data = pd.read_csv(nome_ficheiro_dados_novos, header=None)

try:
    res = np.loadtxt(nome_ficheiro_resultados)
except Exception as e:
    print(f"ERRO: {e}")
    erros=True

if not erros:
    
    linha = 0
    ID_grupo = res[linha]
    if not ID_grupo==GRUPO:
        print("ERRO: ID do grupo no ficheiro diferente da variável GRUPO.")
        erros = True
    if ID_grupo<1 or ID_grupo>99:
        print("ERRO: ID do grupo deve ser entre 1 e 99.")
        erros = True
    if not ID_grupo.is_integer():
        print("ERRO: ID do grupo deve ser um inteiro.")
        erros = True
    linha += 1

    DT_Accuracy = res[linha:linha+n_medidas_estimativa]
    if not np.all((DT_Accuracy >= 0) & (DT_Accuracy <= 1)):
        print("ERRO (DT): Métricas de Accuracy devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_DT = [DT_Accuracy[6],DT_Accuracy[4],DT_Accuracy[0],DT_Accuracy[1],DT_Accuracy[2],DT_Accuracy[3],DT_Accuracy[5]]
        if not (sorted_DT == sorted(DT_Accuracy)):
            print("ERRO (DT): Métricas de Accuracy estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    DT_F1 = res[linha:linha+n_medidas_estimativa]
    if not np.all((DT_F1 >= 0) & (DT_F1 <= 1)):
        print("ERRO (DT): Métricas de F1 devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_DT = [DT_F1[6],DT_F1[4],DT_F1[0],DT_F1[1],DT_F1[2],DT_F1[3],DT_F1[5]]
        if not (sorted_DT == sorted(DT_F1)):
            print("ERRO (DT): Métricas de F1 estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    DT_MCC = res[linha:linha+n_medidas_estimativa]
    if not np.all((DT_MCC >= -1) & (DT_MCC <= 1)):
        print("ERRO (DT): Métricas de MCC devem ser todas entre -1 e 1.")
        erros = True
    else:
        sorted_DT = [DT_MCC[6],DT_MCC[4],DT_MCC[0],DT_MCC[1],DT_MCC[2],DT_MCC[3],DT_MCC[5]]
        if not (sorted_DT == sorted(DT_MCC)):
            print("ERRO (DT): Métricas de MCC estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    DT_previsoes = res[linha:linha+data.shape[0]]
    if not np.all((DT_previsoes==0) | (DT_previsoes==1)):
        print("ERRO (DT): Previsões das classes devem todas ser 0 ou 1.")
        erros = True
    linha += data.shape[0]

    KNN_Accuracy = res[linha:linha+n_medidas_estimativa]
    if not np.all((KNN_Accuracy >= 0) & (KNN_Accuracy <= 1)):
        print("ERRO (KNN): Métricas de Accuracy devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_KNN = [KNN_Accuracy[6],KNN_Accuracy[4],KNN_Accuracy[0],KNN_Accuracy[1],KNN_Accuracy[2],KNN_Accuracy[3],KNN_Accuracy[5]]
        if not (sorted_KNN == sorted(KNN_Accuracy)):
            print("ERRO (KNN): Métricas de Accuracy estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    KNN_F1 = res[linha:linha+n_medidas_estimativa]
    if not np.all((KNN_F1 >= 0) & (KNN_F1 <= 1)):
        print("ERRO (KNN): Métricas de F1 devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_KNN = [KNN_F1[6],KNN_F1[4],KNN_F1[0],KNN_F1[1],KNN_F1[2],KNN_F1[3],KNN_F1[5]]
        if not (sorted_KNN == sorted(KNN_F1)):
            print("ERRO (KNN): Métricas de F1 estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    KNN_MCC = res[linha:linha+n_medidas_estimativa]
    if not np.all((KNN_MCC >= -1) & (KNN_MCC <= 1)):
        print("ERRO (KNN): Métricas de MCC devem ser todas entre -1 e 1.")
        erros = True
    else:
        sorted_KNN = [KNN_MCC[6],KNN_MCC[4],KNN_MCC[0],KNN_MCC[1],KNN_MCC[2],KNN_MCC[3],KNN_MCC[5]]
        if not (sorted_KNN == sorted(KNN_MCC)):
            print("ERRO (KNN): Métricas de MCC estão numa ordem errada.")
            erros = True    
    linha += n_medidas_estimativa
    KNN_previsoes = res[linha:linha+data.shape[0]]
    if not np.all((KNN_previsoes==0) | (KNN_previsoes==1)):
        print("ERRO (KNN): Previsões das classes devem todas ser 0 ou 1.")
        erros = True
    linha += data.shape[0]

    NB_Accuracy = res[linha:linha+n_medidas_estimativa]
    if not np.all((NB_Accuracy >= 0) & (NB_Accuracy <= 1)):
        print("ERRO (NB): Métricas de Accuracy devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_NB = [NB_Accuracy[6],NB_Accuracy[4],NB_Accuracy[0],NB_Accuracy[1],NB_Accuracy[2],NB_Accuracy[3],NB_Accuracy[5]]
        if not (sorted_NB == sorted(NB_Accuracy)):
            print("ERRO (NB): Métricas de Accuracy estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    NB_F1 = res[linha:linha+n_medidas_estimativa]
    if not np.all((NB_F1 >= 0) & (NB_F1 <= 1)):
        print("ERRO (NB): Métricas de F1 devem ser todas entre 0 e 1.")
        erros = True
    else:
        sorted_NB = [NB_F1[6],NB_F1[4],NB_F1[0],NB_F1[1],NB_F1[2],NB_F1[3],NB_F1[5]]
        if not (sorted_NB == sorted(NB_F1)):
            print("ERRO (NB): Métricas de F1 estão numa ordem errada.")
            erros = True
    linha += n_medidas_estimativa
    NB_MCC = res[linha:linha+n_medidas_estimativa]
    if not np.all((NB_MCC >= -1) & (NB_MCC <= 1)):
        print("ERRO (NB): Métricas de MCC devem ser todas entre -1 e 1.")
        erros = True
    else:
        sorted_NB = [NB_MCC[6],NB_MCC[4],NB_MCC[0],NB_MCC[1],NB_MCC[2],NB_MCC[3],NB_MCC[5]]
        if not (sorted_NB == sorted(NB_MCC)):
            print("ERRO (NB): Métricas de MCC estão numa ordem errada.")
            erros = True
    linha += data.shape[0]

if not erros:
    print('OK!')