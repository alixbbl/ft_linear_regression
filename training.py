import csv, argparse, os
import pandas as pd
import DataFrame

# on utilise le fichier data.csv en input
# Ce programme renvoie un output avec les deux valeurs issues du rain theta 0 et theta 1.

def is_valid_path(filepath: str)-> bool:
    try :
        return os.path.exists(filepath)
    except Exception as e:
        return False, print(f'Something happened{e}')

def ft_train(dataframe: DataFrame)-> None:
    theta0=0
    theta1=0
    learningRate=0.01 # valeur standard qu'on peut ajuster

def main(argpath):
    if is_valid_path(argpath):
        dataframe=pd.read_csv(argpath)
    print(dataframe)
    ft_train(dataframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file',\
                        type=str,\
                        help="Please enter a CSV file...")
    argpath=parser.parse_args()
    main(argpath.csv_file)