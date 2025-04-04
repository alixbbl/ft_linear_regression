import os
import pandas as pd
import numpy as np
 
# ********************* LOGIQUE PARSING ****************************
 
def is_valid_path(filepath: str)-> bool:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Error: file {filepath} cannot be found.')
    return True
 
def is_valid_dataframe(df: pd.DataFrame)-> bool:
    col = df.columns.tolist()
    if len(col) != 2:
        raise Exception(f'Dataframe must contains exactly 2 columns!')
    if col[0] is None or col[1] is None:
        raise ValueError(f'Submitted dataframe is incorrect (columns).')
    if not df.map(lambda x: isinstance(x, int)).all().all():
        raise ValueError(f'All data must be integer.')
    return True
 
# *************** LOGIQUE RESHAPING & STANDARDIZATION ******************
 
# Pour etre utilise pr les libs telles que NumPy Dot, ou sklearn, on utilise
# des vecteurs 2D, or ici, y et x sont a une dimension (series ou listes).
# On va donc reshape pour les rendre utilisables.
def prepare_data(df: pd.DataFrame):
 
    x = np.array(df.iloc[:, 0].values).reshape(-1, 1) # on receupere les valeurs interessantes
    y = np.array(df.iloc[:, 1].values).reshape(-1, 1)
    nx = np.zeros(len(x), dtype=float).reshape(-1, 1)
 
    # PHASE DE NORMALISATION : 
    # Pour eviter que les "outliers" ne perturbent le modele on "normalise" les km
    xmin = np.min(x)
    xmax = np.max(x)
    # On recalcule la matrice des km mais normalisee
    for i in range(len(x)):
        nx[i] = (x[i]-xmin)/(xmax-xmin)
    X = np.hstack((nx, np.ones(nx.shape)))
   
    return X, y, xmin, xmax