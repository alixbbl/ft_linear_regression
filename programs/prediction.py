import argparse

# on utilise la partie prediction une fois obtenues les valeurs de theta 0 et theta 1. 
# si l'emtrainment n'a pas eu lie, on les set a 0.
# ces valeurs doivent etre extraites d'un fichier "model.txt" cree par train.py

def ft_predict(mileage: int)-> float:
    pass

def main(args):
    print(f'You required a prediction for {args.mileage} km.')
    prediction=ft_predict(args.mileage)
    print(f'... Corresponds to a {prediction} euros !')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mileage',\
                        type=int,\
                        help="Please enter a valid mileage...")
    parser.add_argument('file_model',\
                        type=int,\
                        help="Corresponds to the path of the model file.")
    args=parser.parse_args()
    main(args)