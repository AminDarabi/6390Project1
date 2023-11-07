# Developed by: Amin Darabi
"""
    This file contains Second method for IFT6390B Data Competition.
"""
from ast import literal_eval
from datetime import date
import argparse

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np


def read_climate_data(
        data_path: str,
        drop_duplicates: bool,
        year_include: list = None,
        year_exclude: list = None,
        add_wind_inverses: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """
    Read climate data from the given path.

    Parameters
    ----------
    data_path : str
        Path to the data.
    drop_duplicates : bool
        Whether to drop duplicate rows or not.
    year_include : list
        List of years to include.
    year_exclude : list
        List of years to exclude.
    add_wind_inverses : bool
        Whether to add inverses of wind speeds or not.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Data and labels.
    """

    data = pd.read_csv(data_path, index_col=0)

    if drop_duplicates:
        data = data.drop_duplicates()

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

    if year_include is not None:
        data = data[data['year'].isin(year_include)]
    if year_exclude is not None:
        data = data[~data['year'].isin(year_exclude)]

    data = data.drop(columns=['time', 'year', 'month', 'day'])

    if add_wind_inverses:

        data['U850inv'] = data['U850'].apply(lambda x: x if x < 0 else 0)
        data['U850'] = data['U850'].apply(lambda x: x if x > 0 else 0)

        data['V850inv'] = data['V850'].apply(lambda x: x if x < 0 else 0)
        data['V850'] = data['V850'].apply(lambda x: x if x > 0 else 0)

        data['UBOTinv'] = data['UBOT'].apply(lambda x: x if x < 0 else 0)
        data['UBOT'] = data['UBOT'].apply(lambda x: x if x > 0 else 0)

        data['VBOTinv'] = data['VBOT'].apply(lambda x: x if x < 0 else 0)
        data['VBOT'] = data['VBOT'].apply(lambda x: x if x > 0 else 0)

    data = data[sorted(data.columns)]
    if 'Label' in data.columns:
        data_x = data.drop('Label', axis=1).astype('float32')
        data_y = data['Label'].astype('int8')
    else:
        data_x = data.astype('float32')
        data_y = None

    return data_x, data_y


def main(args):
    """
    Main function.
    """

    args.layers = literal_eval(args.layers)
    if args.hyperparameter_tuning:

        train_x, train_y = read_climate_data(
            args.train_data, False, year_exclude=[2006, 2007, 2008, 2009])
        validation_x, validation_y = read_climate_data(
            args.train_data, False, year_include=[2006, 2007, 2008, 2009])

        alphas = [0.0001, 0.001, 0.01, 0.1]
        layers = [
            (2,), (3,), (4,), (5,), (6,), (8,), (16,),
            (3, 2), (4, 2), (6, 2), (8, 4), (4, 4), (4, 3),
            (4, 4, 2)
        ]
        best_acc = -1

        for alpha in alphas:
            for layer in layers:

                model = Pipeline([
                    ('Scaler', MinMaxScaler()),
                    ('MLP', MLPClassifier(
                        hidden_layer_sizes=layer, activation=args.activation,
                        max_iter=args.epochs, solver=args.solver,
                        alpha=alpha, random_state=6390
                    ))
                ]).fit(train_x, train_y)

                train_loss = log_loss(train_y, model.predict_proba(train_x))
                train_accuracy = model.score(train_x, train_y)
                valid_loss = log_loss(
                    validation_y, model.predict_proba(validation_x))
                valid_accuracy = model.score(validation_x, validation_y)

                if valid_accuracy > best_acc:
                    best_acc = valid_accuracy
                    args.alpha = alpha
                    args.layers = layer

                if args.print:
                    print('-' * 60)
                    print(f"regularization rate: {alpha}\t layer: {layer}")
                    print(
                        f"train accuracy: {train_accuracy}\t"
                        f"Train loss: {train_loss}"
                    )
                    print(
                        f"validation accuracy: {valid_accuracy}"
                        f"\tValidation loss: {valid_loss}"
                    )

        if args.print:
            print('=' * 60)
            print(
                f"best regularization rate: {args.alpha}\t"
                f"best layer size: {args.layers}")
            print(f"best validation accuracy: {best_acc}")
            print('=' * 60)

    if args.validate:

        train_x, train_y = read_climate_data(
            args.train_data, False, year_exclude=[2008, 2009])
        validation_x, validation_y = read_climate_data(
            args.train_data, False, year_include=[2008, 2009])

        model = Pipeline([
            ('Scaler', MinMaxScaler()),
            ('MLP', MLPClassifier(
                hidden_layer_sizes=args.layers, activation=args.activation,
                max_iter=args.epochs, solver=args.solver,
                alpha=args.alpha, random_state=6390
            ))
        ]).fit(train_x, train_y)

        print(
            f"Train: {log_loss(train_y, model.predict_proba(train_x))}, "
            f"{model.score(train_x, train_y)}"
        )
        print(
            "Valid:"
            f" {log_loss(validation_y, model.predict_proba(validation_x))}, "
            f"{model.score(validation_x, validation_y)}"
        )

    train_x, train_y = read_climate_data(args.train_data, False)
    test, _ = read_climate_data(args.test_data, False)

    model = Pipeline([
        ('Scaler', MinMaxScaler()),
        ('MLP', MLPClassifier(
            hidden_layer_sizes=args.layers, activation=args.activation,
            verbose=args.print and not args.hyperparameter_tuning,
            max_iter=args.epochs, solver=args.solver,
            alpha=args.alpha, random_state=6390
        ))
    ]).fit(train_x, train_y)

    if args.hyperparameter_tuning or args.validate or args.print:
        print(
            f"Train: {log_loss(train_y, model.predict_proba(train_x))}, "
            f"{model.score(train_x, train_y)}"
        )

    test['Label'] = model.predict(test)
    test['Label'].to_csv(args.output, header=True)

    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Logistic Regression Classifier',
        description="Logistic Regression Classifier."
                    "This program is written as a baseline method "
                    "for IFT6390B Data Competition. "
                    "Developed by: Amin Darabi october 2023."
    )
    parser.add_argument('train_data', type=str, help="Path to the train data")
    parser.add_argument('test_data', type=str, help="Path to the test data")
    parser.add_argument('-v', '--validate', action='store_true',
                        help="Validate the model")
    parser.add_argument('-o', '--output', type=str, default='output.csv',
                        help="Path to the output file")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="Regularization rate")
    parser.add_argument('-ep', '--epochs', type=int, default=1000,
                        help='Number of maximum iterations')
    parser.add_argument('-p', '--print', action='store_true',
                        help="Print loss and accuracy during training and hps")
    parser.add_argument('-hps', '--hyperparameter_tuning', action='store_true',
                        help="Select learning rate and regularization rate "
                        "automatically")
    parser.add_argument('--activation', type=str, default='relu',
                        help="Activation function")
    parser.add_argument('--solver', type=str, default='adam',
                        help="Solver to use")
    parser.add_argument('--layers', type=str, default='(4, 2)',
                        help="Hidden layer sizes")
    main(parser.parse_args())
