import argparse
import numpy as np
import sys, os
 
# on utilise la partie prediction une fois obtenues les valeurs de theta 0 et theta 1.
# si l'emtrainment n'a pas eu lie, on les set a 0.
# ces valeurs doivent etre extraites d'un fichier "model.txt" cree par train.py
 
# on recupere les valeurs de theta et on procede au calcul matriciel avec la donnee fournie par
# le user, convertie au format de matrice (1, mileage)
# on multiplie le 1 avec l'intercept
def ft_predict(theta: np.ndarray, mileage: int)-> float:
    X=np.array([1, mileage]).reshape(1, -1)
    prediction=X.dot(theta)
    return prediction.item()
 
 
def main(args):
 
    if args.mileage is None:
        mileage=0
    else:
        mileage=abs(args.mileage)
    print(f'You required a prediction for {mileage} km.')
    theta=np.zeros(2)
    if os.path.exists(args.input):
        with open(args.input, 'r') as input:
            theta_lines=input.readlines()
            print(f'Contenu du file : {theta_lines}')
            print(len(theta_lines))
            if len(theta_lines) == 2:
                try:
                    theta[0]=float(theta_lines[0].strip())
                    theta[1]=float(theta_lines[1].strip())
                    print(theta[0])
                    print(theta[1])
                except Exception as e:
                    raise("Corrupted data in the theta file.")
            else:
                print('AHAHAHAH')
    else:
        print("This file doesn't exist...")
    print(theta)
    prediction=ft_predict(theta, mileage)
    print(f'... Corresponds to a {prediction:.4f} euros !')
 
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('mileage',\
                        type=int,\
                        help="Please enter a valid mileage...")
    parser.add_argument('input',\
                        type=str,\
                        help="Corresponds to the path of the theta file.")
    args=parser.parse_args()
    main(args)