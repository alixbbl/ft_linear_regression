import csv, argparse, os
import pandas as pd

# on utilise le fichier data.csv en input
# Ce programme renvoie un output avec les deux valeurs issues du rain theta 0 et theta 1.

# ******************* LOGIQUE PARSING **************************

def is_valid_path(filepath: str)-> bool:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Error: file {filepath} cannot be found.')
    return True

def is_valid_dataframe(df: pd.DataFrame)-> bool:
    col=df.columns.tolist()
    if len(col) != 2:
        raise Exception(f'Dataframe must contains exactly 2 columns!')
    if col[0] != 'km' or col[1] != 'price':
        raise ValueError(f'Submitted dataframe is incorrect (columns).')
    if not df.applymap(lambda x: isinstance(x, int)).all().all():
        raise ValueError(f'All data must be integer.')

# *************** LOGIQUE REGRESSION LINEAIRE ******************

def ft_train(dataframe: pd.DataFrame)-> None:
    theta0=0
    theta1=0
    learningRate=0.01 # valeur standard qu'on peut ajuster


# ******************* MAIN ET PARSER **************************

def main(argpath):
    if is_valid_path(argpath):
        try:
            dataframe=pd.read_csv(argpath)
            # print(dataframe)
            if is_valid_dataframe(dataframe):
                try:
                    ft_train(dataframe)
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