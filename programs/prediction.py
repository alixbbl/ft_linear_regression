import argparse
import numpy as np
import pandas as pd
import sys, os
from utils.prepare_data import is_valid_path
import matplotlib.pyplot as plt

# ************************************ COMMENTS **********************************************

# on utilise la partie prediction une fois obtenues les valeurs de theta 0 et theta 1.
# si l'emtrainment n'a pas eu lie, on les set a 0.
# ces valeurs doivent etre extraites d'un fichier "thetas.txt" cree par training.py
# on recupere les valeurs de theta et on procede au calcul matriciel avec la donnee fournie par
# le user, convertie au format de matrice (1, mileage)
# on multiplie le 1 avec l'intercept


# ************************************** CODE ***********************************************

def ft_predict(theta: np.ndarray, mileage: int)-> float:
    X=np.array([1, mileage]).reshape(1, -1)
    prediction=X.dot(theta)
    return prediction.item()
 
def main(args):
    try:
        mileage = 0
        while True:
            
            try: 
                mileage = int(input('Please enter a realistic mileage : '))
                mileage = abs(mileage)
                print("Thanks!")
                break
            
            except ValueError:
                print("This won't work for me buddy ... Please enter a valid integer.")
        
        theta=np.zeros(2) 
        if os.path.exists(args.thetas_input):
            with open(args.thetas_input, 'r') as thetas_input:
                theta_lines=thetas_input.readlines()
                if len(theta_lines) == 2:
                    try:
                        theta[0] = float(theta_lines[0].strip())
                        theta[1] = float(theta_lines[1].strip())
                        print(theta[0])
                        print(theta[1])

                    except Exception as e:
                        raise("Corrupted data in the theta file.")
                else:
                    print('Some issues with thetas file.')
        else:
            print("This file doesn't exist...")

        prediction = ft_predict(theta, mileage)
        print(f'... Corresponds to a {prediction:.4f} euros !')
        
        if is_valid_path(args.data_for_graphic):
            df =  pd.read_csv(args.data_for_graphic)
        else:
            raise Exception('This filepath does not exist !')
        
        X = np.array(df['km']).reshape(-1, 1)  # X en majuscule pour les kilomètres (données d'entrée)
        y = np.array(df['price']).reshape(-1, 1)
        plt.scatter(X, y, color='blue', label='Data Points')  # Points de données réelles

        # trace la droite de régression : y = theta0 + theta1 * x
        x_values = np.linspace(min(X), max(X), 100)
        y_values = theta[0] + theta[1] * x_values
        plt.plot(x_values, y_values, color='red', label='Regression Line')
        # pour l'affichage
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price (Euros)')
        plt.title('Linear Regression: Price vs Mileage')
        plt.legend()
        plt.show()
    
    except KeyboardInterrupt:
       print('Oh, you jus press CTRL+C, bye bye !')
       sys.exit(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('thetas_input',\
                        type = str,\
                        help = "Corresponds to the path of the theta file.")
    parser.add_argument('data_for_graphic',\
                        type = str,\
                        default="./data/data.csv",
                        help = "Corresponds to the path of the data file, used for graphic visualization.")
    args = parser.parse_args()
    main(args)