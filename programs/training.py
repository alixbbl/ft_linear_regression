import pandas as pd
import numpy as np
import argparse, os
from utils.prepare_data import is_valid_path, is_valid_dataframe, prepare_data


# *************** LOGIQUE REGRESSION LINEAIRE ******************

# Attention : on travaille en objet NumPy et pas en Df...
# On retourne simplement le produit matriciel des deux np
def model(X: np.ndarray, theta: np.ndarray)-> np.ndarray:
    return X.dot(theta)

# def training(theta0=0, theta1=0):
#     theta=np.array([[theta0], [theta1]])

# il s'agit de la MSE/2, on peut aussi utiliser la MAE => tester 
def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray)-> None:
    m=len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)


# ******************* MAIN ET PARSER **************************

def main(argpath):
    if is_valid_path(argpath):
        try:
            dataframe=pd.read_csv(argpath)
            if is_valid_dataframe(dataframe):
                try:
                    prepare_data(dataframe)
                except Exception as e:
                    print(f'{e}')
        except Exception as e:
            print(f'{e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file',\
                        type=str,\
                        help="Please enter a CSV file...")
    argpath=parser.parse_args()
    main(argpath.csv_file)