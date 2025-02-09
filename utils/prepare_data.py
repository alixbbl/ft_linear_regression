import os
import pandas as pd
import numpy as np

# ********************* LOGIQUE PARSING ****************************

def is_valid_path(filepath: str)-> bool:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Error: file {filepath} cannot be found.')
    return True

def is_valid_dataframe(df: pd.DataFrame)-> bool:
    col=df.columns.tolist()
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
def prepare_data(df: pd.DataFrame)-> pd.DataFrame:
    
    x=df.iloc[:, 0]
    y=df.iloc[:, 1]
    # print(f'Original shape of x is : {x.shape}')
    # print(f'Original shape of y is : {y.shape}')

    x=x.values.reshape(-1, 1) # -1 signifie pour Python "mets autant de lignes que necessaire"
    y=y.values.reshape(-1, 1)
    # print(f'Final shape of x is : {x.shape}')
    # print(f'Final shape of y is : {y.shape}')

    # Standardization : 
    # obligatoire en Regression Lineaire pour eviter que le modele ne favorise les
    # valeurs les plus grandes, on "lisse"
    x_mean=x.mean()
    x_std=x.std()
    x_standardized=(x - x_mean) / x_std

    # Mise en place de la matrice des features (x1, x2, x3 ...), avec deux colonnes,
    # en effet, on prevoit pour chaque sample la place pour le theta0 (le b).
    X=np.hstack((x_standardized, np.ones(x.shape)))
    # print(X)
