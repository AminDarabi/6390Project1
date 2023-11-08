# Developed by: Amin Darabi
"""
    This file contains Baseline model for IFT6390 Data Competition.
"""
from datetime import date
import argparse

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


class LogisticRegression:
    """
    Logistic Regression model.

    Methods
    -------
    __init__(
        in_dimension: int,
        out_dimension: int,
        regularization: float = 1.0,
        initial_weights: np.ndarray = None,
        initial_bias: np.ndarray = None,
        scaler: str = 'MinMax',
        solver: str = 'dummy')
        Initialize the model.
    fit(
        train_x: np.ndarray,
        train_y: np.ndarray,
        max_epoch: int = 200,
        print_loss: bool = False) -> LogisticRegression
        Fit the model to the train data.
    predict(data_x: np.ndarray) -> np.ndarray
        Predict the labels for the input data.
    validate(
        validate_x: np.ndarray,
        validate_y: np.ndarray) -> Tuple[float, float]
        Validate the model.

    Attributes
    ----------
    regularization : float
        Regularization strength.
    scaler_method : str
        Scaler method.

    Notes
    -----
    list of solvers:
        'dummy'
        'GradientDescent'
    list of scalers:
        'MinMax'
        'AbsMax'
    """

    _solver_list = ['dummy', 'GradientDescent']
    _scaler_list = ['MinMax', 'AbsMax']

    def __init__(
            self, in_dimension: int,
            out_dimension: int,
            regularization: float = 1.0,
            initial_weights: np.ndarray = None,
            initial_bias: np.ndarray = None,
            scaler: str = 'MinMax',
            solver: str = 'dummy'):
        """
        Initialize the model.
        """

        if out_dimension < 1 or in_dimension < 1:
            raise ValueError("in_dimension and out_dimension must be positive")

        # initializing weights
        if not isinstance(initial_weights, np.ndarray):
            self._weights = np.random.uniform(
                -1, 1, (in_dimension, out_dimension)).astype('float32')
        elif initial_weights.shape == (in_dimension, out_dimension):
            self._weights = initial_weights.astype('float32')
        else:
            raise ValueError("initial_weights has wrong shape")

        # initializing bias
        if not isinstance(initial_bias, np.ndarray):
            self._bias = np.random.uniform(-1, 1, (out_dimension,)).astype(
                'float32')
        elif initial_bias.shape == (out_dimension,):
            self._bias = initial_bias.astype('float32')
        else:
            raise ValueError("initial_bias has wrong shape")

        # initializing regularization strength
        self.regularization = regularization

        # initializing scaler
        if scaler not in self._scaler_list:
            raise ValueError(f"Scaler must be one of {self._scaler_list}")
        self.scaler_method = scaler
        self._scaler = None

        # initializing solver
        if solver not in self._solver_list:
            raise ValueError(f"Solver must be one of {self._solver_list}")
        self._solver = solver

    def __dummy_solve(
            self,
            data_x: np.ndarray,
            data_y: np.ndarray,
            max_epoch: int = 200,
            print_loss: bool = False):
        # This function is a dummy solver

        weights_rates = np.zeros(self._weights.shape).astype('float32')
        weights_rates[:, :] = 1e-5
        bias_rates = np.zeros(self._bias.shape).astype('float32')
        bias_rates[:] = 1e-5
        last_weights_sign = np.ones(self._weights.shape).astype('int8')
        last_bias_sign = np.ones(self._bias.shape).astype('int8')

        for epoch in range(max_epoch):

            predict = self._predict(data_x)
            gradient, gradient_bias = self._gradient(data_x, predict, data_y)

            gradient_sign = np.sign(gradient).astype('int8')
            gradient_bias_sign = np.sign(gradient_bias).astype('int8')

            weights_rates = np.where(
                gradient_sign == last_weights_sign,
                weights_rates * 1.25,
                weights_rates * 0.5
            )
            bias_rates = np.where(
                gradient_bias_sign == last_bias_sign,
                bias_rates * 1.25,
                bias_rates * 0.5
            )

            last_weights_sign = gradient_sign
            last_bias_sign = gradient_bias_sign

            self._weights -= weights_rates * gradient
            self._bias -= bias_rates * gradient_bias

            if print_loss:
                print(f"Epoch {epoch + 1}:\t{self._loss(predict, data_y)}")

    def _gradient_descent(
            self,
            data_x: np.ndarray,
            data_y: np.ndarray,
            max_epoch: int,
            print_loss: bool,
            learning_rate: float = 1e-4):
        # This function is a gradient descent solver

        for epoch in range(max_epoch):

            predict = self._predict(data_x)
            gradient, gradient_bias = self._gradient(data_x, predict, data_y)

            self._weights -= learning_rate * gradient
            self._bias -= learning_rate * gradient_bias

            if print_loss:
                print(f"Epoch {epoch + 1}:\t{self._loss(predict, data_y)}")

    def _solve(
            self,
            data_x: np.ndarray,
            data_y: np.ndarray,
            max_epoch: int,
            print_loss: bool):
        # This function finds the best weights and bias
        if self._solver == 'dummy':
            self.__dummy_solve(
                data_x, data_y, max_epoch, print_loss)
        elif self._solver == 'GradientDescent':
            self._gradient_descent(
                data_x, data_y, max_epoch, print_loss)
        else:
            raise ValueError(f"Solver must be one of {self._solver_list}")

    def _set_scaler(self, train_x: np.ndarray):
        # This function sets the scaler
        if self.scaler_method == 'MinMax':
            self._scaler = self._MinMaxScaler(train_x)
        elif self.scaler_method == 'AbsMax':
            self._scaler = self._AbsMaxScaler(train_x)
        elif self.scaler_method == 'standard':
            self._scaler = self._StandardScaler(train_x)
        else:
            raise ValueError(f"Scaler must be one of {self._scaler_list}")

    def _predict(self, data_x: np.ndarray) -> np.ndarray:
        # This function predicts the labels for the input data
        predict = np.exp(data_x @ self._weights + self._bias)
        predict = predict / predict.sum(1)[:, None]
        return predict

    def _gradient(
            self,
            data_x: np.ndarray,
            predict_y: np.ndarray,
            data_y: np.ndarray) -> np.ndarray:
        # This function calculates the gradient
        gradient = data_x.T @ (predict_y - data_y) +\
            self.regularization * self._weights
        gradient_bias = np.sum(predict_y - data_y) +\
            self.regularization * self._bias
        return gradient, gradient_bias

    def _process_data_x(self, data_x: np.ndarray) -> np.ndarray:
        # This function processes the input data
        if not isinstance(data_x, np.ndarray):
            data_x = np.array(data_x)
        if data_x.shape[1] != self._weights.shape[0]:
            raise ValueError("data_x has wrong shape")
        return data_x.astype('float32')

    def _process_data_y(self, data_y: np.ndarray) -> np.ndarray:
        # This function processes the input labels
        if not isinstance(data_y, np.ndarray):
            data_y = np.array(data_y)
        if data_y.shape in [(data_y.shape[0],), (data_y.shape[0], 1)]:
            data_y = (data_y == np.unique(data_y)[:, None]).T
        if data_y.shape[1] != self._weights.shape[1]:
            raise ValueError("data_y has wrong shape")
        return data_y.astype('float32')

    @staticmethod
    def _loss(predict: np.ndarray, data_y: np.ndarray) -> float:
        # This function calculates the mean of the cross-entropy loss
        return -(np.log(predict) * data_y).sum() / data_y.shape[0]

    @staticmethod
    def _accuracy(predict: np.ndarray, data_y: np.ndarray) -> float:
        # This function calculates the accuracy
        return (predict.argmax(1) == data_y.argmax(1)).sum() / len(data_y)

    def fit(self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            max_epoch: int,
            print_loss: bool):
        """
        Fit the model to the train data.

        Parameters
        ----------
        train_x : np.ndarray
            Train data.
            shape: (n, in_dimension)
        train_y : np.ndarray
            Train labels.
            shape: (n, out_dimension) or (n,)
        max_epoch : int
            Maximum number of epochs to train the model.
        print_loss : bool
            Whether to print loss and accuracy during training or not.

        Returns
        -------
        LogisticRegression
            self.

        Raises
        ------
        ValueError
            If the input data has wrong shape.
            train_x and validate_x must have shape (n, in_dimension).
            train_y and validate_y must have shape (n, out_dimension).
        """

        train_x = self._process_data_x(train_x)
        train_y = self._process_data_y(train_y)
        self._set_scaler(train_x)
        train_x = self._scaler.transform(train_x)

        self._solve(train_x, train_y, max_epoch, print_loss)
        return self

    def predict(self, data_x: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data.

        Parameters
        ----------
        data_x : np.ndarray
            Input data.
            shape: (n, in_dimension)

        Returns
        -------
        np.ndarray
            Predicted labels.

        Raises
        ------
        ValueError
            If the input data has wrong shape.
            data_x must have shape (n, in_dimension).
        """
        data_x = self._process_data_x(data_x)
        data_x = self._scaler.transform(data_x)
        return self._predict(data_x).argmax(1).astype('int8')

    def validate(
            self,
            validate_x: np.ndarray,
            validate_y: np.ndarray) -> (float, float):
        """
        Validate the model.

        Parameters
        ----------
        validate_x : np.ndarray
            Validation data.
            shape: (m, in_dimension)
        validate_y : np.ndarray
            Validation labels.
            shape: (m, out_dimension) or (m,)

        Returns
        -------
        Tuple[float, float]
            (loss, accuracy)
        """

        validate_x = self._process_data_x(validate_x)
        validate_y = self._process_data_y(validate_y)
        validate_x = self._scaler.transform(validate_x)

        predict = self._predict(validate_x)
        return (
            self._loss(predict, validate_y),
            self._accuracy(predict, validate_y)
        )

    class _MinMaxScaler:
        """
        Min-Max Scaler class that Scale features between -1 and +1.

        Methods
        -------
        transform(data: np.ndarray) -> np.ndarray
            Transform the input data.
        """

        def __init__(self, data: np.ndarray):
            """
            Initialize the scaler.

            Parameters
            ----------
            data : np.ndarray
                Data to fit the scaler.
            """

            self._min = data.min(0)
            self._max = data.max(0)

        def transform(self, data: np.ndarray) -> np.ndarray:
            """
            Transform the input data.

            Parameters
            ----------
            data : np.ndarray
                Data to transform.

            Returns
            -------
            np.ndarray
                Transformed data.
            """
            return ((data - self._min) / (self._max - self._min)) * 2 - 1

    class _AbsMaxScaler:
        """
        Abs-Max Scaler class that Scale features between -1 and +1.

        Methods
        -------
        transform(data: np.ndarray) -> np.ndarray
            Transform the input data.
        """

        def __init__(self, data: np.ndarray):
            """
            Initialize the scaler.

            Parameters
            ----------
            data : np.ndarray
                Data to fit the scaler.
            """

            self._max = np.abs(data).max(0)

        def transform(self, data: np.ndarray) -> np.ndarray:
            """
            Transform the input data.

            Parameters
            ----------
            data : np.ndarray
                Data to transform.

            Returns
            -------
            np.ndarray
                Transformed data.
            """
            return data / self._max

    class _StandardScaler:
        """
        Standard Scaler class that Scale features to have mean 0 and std 1.

        Methods
        -------
        transform(data: np.ndarray) -> np.ndarray
            Transform the input data.
        """

        def __init__(self, data: np.ndarray):
            """
            Initialize the scaler.

            Parameters
            ----------
            data : np.ndarray
                Data to fit the scaler.
            """

            self._mean = data.mean(0)
            self._std = data.std(0)

        def transform(self, data: np.ndarray) -> np.ndarray:
            """
            Transform the input data.

            Parameters
            ----------
            data : np.ndarray
                Data to transform.

            Returns
            -------
            np.ndarray
                Transformed data.
            """
            return (data - self._mean) / self._std


def main(args):
    """
    Main function.
    """

    if args.hyperparameter_tuning:

        train_x, train_y = read_climate_data(
            args.train_data, False, year_exclude=[2006, 2007, 2008, 2009])
        validation_x, validation_y = read_climate_data(
            args.train_data, False, year_include=[2006, 2007, 2008, 2009])

        regularization_rates = [0.1, 0.2, 0.5, 1.0, 2.0]
        max_epochs = [200, 400, 800]
        best_loss = 1000000.0

        for rg in regularization_rates:
            for ep in max_epochs:

                model = LogisticRegression(
                    train_x.shape[1],
                    len(train_y.unique()),
                    rg,
                    scaler=args.scaler,
                    solver=args.solver
                ).fit(train_x, train_y, ep, False)

                train_loss, train_accuracy = model.validate(train_x, train_y)
                valid_loss, valid_accuracy = model.validate(
                    validation_x, validation_y)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    args.regularization = rg
                    args.epochs = ep

                if args.print:
                    print('-' * 60)
                    print(f"regularization rate: {rg}\t epoch: {ep}")
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
                f"best regularization rate: {args.regularization}\t"
                f"best epoch: {args.epochs}")
            print(f"best validation loss: {best_loss}")
            print('=' * 60)

    if args.validate:

        train_x, train_y = read_climate_data(
            args.train_data, False, year_exclude=[2008, 2009])
        validation_x, validation_y = read_climate_data(
            args.train_data, False, year_include=[2008, 2009])

        model = LogisticRegression(
            train_x.shape[1],
            len(train_y.unique()),
            args.regularization,
            scaler=args.scaler,
            solver=args.solver
        ).fit(train_x, train_y, args.epochs, False)

        print(f"Train: {model.validate(train_x, train_y)}\t")
        print(f"Valid: {model.validate(validation_x, validation_y)}")

    train_x, train_y = read_climate_data(args.train_data, False)
    test, _ = read_climate_data(args.test_data, False)

    model = LogisticRegression(
        train_x.shape[1],
        len(train_y.unique()),
        args.regularization,
        scaler=args.scaler,
        solver=args.solver
    ).fit(
        train_x, train_y, args.epochs,
        args.print and not args.hyperparameter_tuning
    )

    if args.hyperparameter_tuning or args.validate or args.print:
        print(f"Train: {model.validate(train_x, train_y)}")

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
    parser.add_argument('-rg', '--regularization', type=float, default=0.2,
                        help="Regularization rate")
    parser.add_argument('-ep', '--epochs', type=int, default=400,
                        help='Number of epochs')
    parser.add_argument('-p', '--print', action='store_true',
                        help="Print loss and accuracy during training and hps")
    parser.add_argument('-hps', '--hyperparameter_tuning', action='store_true',
                        help="Select learning rate and regularization rate "
                        "automatically")
    parser.add_argument('--solver', type=str, default='dummy',
                        help="Solver method, must be one of "
                        "['dummy', 'GradientDescent']")
    parser.add_argument('--scaler', type=str, default='MinMax',
                        help="Scaler method, must be one of "
                        "['MinMax', 'AbsMax']")
    main(parser.parse_args())
