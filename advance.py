"""
advance method classifier.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd
from sklearn import svm


def read_climate_data(file='data/train.csv') -> pd.DataFrame:
    """
    Read climate data from file.

    Parameters
    ----------
    file : str
        Path to the file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the modified data.
    """

    data = pd.read_csv(file, index_col=0)

    # modifying time in data
    data['time'] = data['time'].astype('str')
    data['year'] = data['time'].apply(lambda x: int(x[:4]))
    data['month'] = data['time'].apply(lambda x: int(x[4:6]))
    data['day'] = data['time'].apply(lambda x: int(x[6:8]))

    data['time'] = data.apply(
        lambda x:
        date(x['year'], x['month'], x['day']).timetuple().tm_yday
        + ((x['year'] - 1) % 4) * 0.2425, axis=1).astype('float32')

    data['SinTime'] = np.sin((2 * np.pi * data['time']) / 365.2425)
    data['CosTime'] = np.cos((2 * np.pi * data['time']) / 365.2425)
    data = data.drop(columns=['time', 'year', 'month', 'day'])

    return data


def main(args):
    """
    Main function.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the program.
    """

    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Logistic Regression Classifier',
        description="Logistic Regression Classifier."
                    "This program is written as an alternative method "
                    "for IFT6390B Data Competition. "
                    "Developed by: Amin Darabi october 2023."
    )
    parser.add_argument('train_data', type=str, help="Path to the train data")
    parser.add_argument('test_data', type=str, help="Path to the test data")
    parser.add_argument('-o', '--output', type=str, default='output.csv',
                        help="Path to the output file")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('-rg', '--regularization', type=float, default=0.1,
                        help="Regularization rate")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('-p', '--print_loss', action='store_true',
                        help="Print loss and accuracy during training")
    parser.add_argument('--not_oversampling', action='store_false',
                        help="Do not over-sample the train data")
    parser.add_argument('-hps', '--hyperparameter_tuning', action='store_true',
                        help="Select learning rate and regularization rate "
                        "automatically")
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                        help="Ratio of validation data to the train data")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle the train set [before splitting]")
    main(parser.parse_args())
