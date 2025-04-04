import pandas as pd
import numpy as np
import argparse, os
from utils.prepare_data import is_valid_path, is_valid_dataframe, prepare_data
import matplotlib.pyplot as plt
 
# ************************ LOGIQUE REGRESSION LINEAIRE **************************
 
# on prend un learning_rate basique de 0.01
# La fonction regroupe les etapes de calcul de predisctions, de l'erreur, du cout pour cette erreur
# puis la mise a jour du
def gradient_descent(X, y, max_iterations=1000, learning_rate=0.1):
 
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_report = []
 
    for i in range(max_iterations):
 
        predictions = X.dot(theta) # on multiplie la matrice des km normalises par le vecteur (0, 0), debut de la descente de gradient
        errors = predictions-y # on calcule l'erreur entre la matrice prediction et la matrice des prix reels du dataset
        gradient = (1/m) * X.T.dot(errors) # formule du gradient
        theta -= learning_rate * gradient # formule d'actualisation de theta (theta0, theta1) l'intercept et la pente
        cost = (1/(2*m)) * np.sum(errors ** 2) # calcul du cout (indicatif de suivi de la descente de gradient)
        cost_report.append(cost)
       
        if i%100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, theta = {theta.T}")
   
    return theta, cost_report
    # Ici, la formule utilise la pente du gradient+learning rate pour donner la "direction"
 
 
# ******************************** MAIN ET PARSER *******************************
 
def main(argpath):
   
    if is_valid_path(argpath):
       
        try:
            dataframe = pd.read_csv(argpath)
            if is_valid_dataframe(dataframe):
               
                try:
                    X, y, xmin, xmax = prepare_data(dataframe)
                    theta_norm, cost_report = gradient_descent(X, y)
                    theta1_real=theta_norm[0, 0] / (xmax - xmin)
                    theta0_real=theta_norm[1, 0] - theta1_real * xmin
                    plt.plot(cost_report)
                    plt.xlabel("Iterations")
                    plt.ylabel("Cost")
                    plt.title("Gradient Descent Progress")
                    plt.grid(True)
                    plt.show()
 
                    theta=np.array([[theta0_real], [theta1_real]])
                    print(f"Theta : {theta}")
 
                    with open('theta.txt', 'w') as file:
                        file.write(f'{theta0_real}\n{theta1_real}')
                        print('Success printing the theta file!')
               
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