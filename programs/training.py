import pandas as pd
import numpy as np
import argparse, os
from utils.prepare_data import is_valid_path, is_valid_dataframe, prepare_data
import matplotlib.pyplot as plt

# ************************ LOGIQUE REGRESSION LINEAIRE **************************

# Attention : on travaille en objet NumPy et pas en Df...
# On retourne simplement le produit matriciel des deux np
def model(X: np.ndarray, theta: np.ndarray)-> np.ndarray:
    return X.dot(theta)

# Mean Square Error (/2) est notre fonction de cout (classique en regression lineaire)
def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray)-> np.ndarray:
    m=len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

# La fonction de gradient est la derivee de la fonction de cout, elle permet de 
# nous indiquer dans quelle direction il faut modifire les parametres de theta
# pour ameliorer la performance de notre modele
def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray)-> None:
    m=len(y)
    predictions=model(X, theta)
    error=predictions - y # calcul de l'erreur
    return (1/m) * X.T.dot(error)

# Fonction de descente de gradient
def gradient_descent(X: np.ndarray, y: np.ndarray, theta0=0, theta1=0,\
                     max_iterations=1000,\
                        learning_rate=0.001)-> np.ndarray:
    theta=np.array([[theta0], [theta1]], dtype=np.float64)
    
    for i in range(0, max_iterations):
        grad=gradient(X, y, theta)
        theta -= learning_rate * grad
        # if i % 100 == 0:
        #     cost = cost_function(X, y, theta)
        #     print(f"Iteration {i}: Cost = {cost:.4f}, theta = {theta.T}")
    return theta
    # Ici, la formule utilise la pente du gradient+learning rate pour donner la "direction"


# ******************************** MAIN ET PARSER *******************************

def main(argpath):
    if is_valid_path(argpath):
        try:
            dataframe=pd.read_csv(argpath)
            if is_valid_dataframe(dataframe):
                try:
                    X, y=prepare_data(dataframe)
                    y_mean=y.mean()
                    y_centered=y - y_mean
                    theta_standardized=gradient_descent(X, y_centered)
                    x_std=dataframe.iloc[:, 0].std()
                    theta0_real = theta_standardized[0, 0]
                    theta1_real = theta_standardized[1, 0] / x_std
                    theta_real = np.array([[theta0_real], [theta1_real]])
                    predictions=model(X, theta_real)
                    x=dataframe.iloc[:, 0]
                    y=dataframe.iloc[:, 1]
                    plt.scatter(x, y)
                    plt.plot(x, predictions, c='r')
                    plt.show()
                    # print(f"Theta0 : {theta_standardized[0, 0]}")
                    # print(f"Theta1 réel (échelle originale) : {theta1_real}")
                    # imprimer un output.txt contenant theta
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