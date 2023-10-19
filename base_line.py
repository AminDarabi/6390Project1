# Developed by: Amin Darabi
"""
    This file contains Baseline model for IFT6390 Data Competition.
"""
from datetime import date
import argparse

import pandas as pd
import numpy as np


def data_split(
        file: str = 'data/train.csv',
        split_ratio: float = 0.85) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the data into train and validation sets.

    Parameters
    ----------
    file : str
        Path to the data file.
    split_ratio : float
        Ratio of train set to the whole data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Train and validation sets.
    """
    data = pd.read_csv(file, index_col='Sno')
    data = data.sample(frac=1).reset_index(drop=True)
    train = data[:int(split_ratio*len(data))]
    validation = data[int(split_ratio*len(data)):]
    return train, validation


def pre_process(
        train: pd.DataFrame | str = 'data/train.csv',
        test: pd.DataFrame | str = 'data/test.csv',
        oversampling: bool = True
        ) -> (pd.DataFrame, pd.DataFrame):
    """
    Pre-process the data.

    Parameters
    ----------
    train : pd.DataFrame
        Train set.
    test : pd.DataFrame
        Test set.
    oversampling : bool
        Whether to oversampling the train set or not.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Pre-processed train and test sets.
    """

    if isinstance(train, str):
        train = pd.read_csv(train, index_col=0)

    if isinstance(test, str):
        test = pd.read_csv(test, index_col=0)

    if test is not None:
        test['Label'] = -1
        test.index += 10000000
        data = pd.concat([train, test])
    else:
        data = train

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

    # modifying location in data
    data['lat'] = (data['lat'] - data['lat'].mean()) / data['lat'].std()
    data['lon'] = (data['lon'] - data['lon'].mean()) / data['lon'].std()

    # modifying temperature in data
    data['T200'] = (data['T200'] - data['T200'].mean()) / data['T200'].std()
    data['T500'] = (data['T500'] - data['T500'].mean()) / data['T500'].std()
    data['TS'] = (data['TS'] - data['TS'].mean()) / data['TS'].std()
    data['TREFHT'] = \
        (data['TREFHT'] - data['TREFHT'].mean()) / data['TREFHT'].std()

    # modifying precipitable water in data
    data['TMQ'] = (data['TMQ'] - data['TMQ'].mean()) / data['TMQ'].std()

    # modifying wind in data
    # FIXME: need a double check
    data['U850'] = data['U850'] / data['U850'].std()
    data['U850inv'] = data['U850'].apply(lambda x: x if x < 0 else 0)
    data['U850'] = data['U850'].apply(lambda x: x if x > 0 else 0)

    data['V850'] = data['V850'] / data['V850'].std()
    data['V850inv'] = data['V850'].apply(lambda x: x if x < 0 else 0)
    data['V850'] = data['V850'].apply(lambda x: x if x > 0 else 0)

    data['UBOT'] = data['UBOT'] / data['UBOT'].std()
    data['UBOTinv'] = data['UBOT'].apply(lambda x: x if x < 0 else 0)
    data['UBOT'] = data['UBOT'].apply(lambda x: x if x > 0 else 0)

    data['VBOT'] = data['VBOT'] / data['VBOT'].std()
    data['VBOTinv'] = data['VBOT'].apply(lambda x: x if x < 0 else 0)
    data['VBOT'] = data['VBOT'].apply(lambda x: x if x > 0 else 0)

    # modifying humidity in data
    data['QREFHT'] = \
        (data['QREFHT'] - data['QREFHT'].mean()) / data['QREFHT'].std()

    # modifying pressure in data
    data['PS'] = (data['PS'] - data['PS'].mean()) / data['PS'].std()
    data['PSL'] = (data['PSL'] - data['PSL'].mean()) / data['PSL'].std()

    # modifying precipitation rate in data
    # FIXME: need a double check
    data['PRECT'] = \
        (data['PRECT'] - data['PRECT'].mean()) / data['PRECT'].std()

    # modifying geopotential height in data
    data['Z1000'] = \
        (data['Z1000'] - data['Z1000'].mean()) / data['Z1000'].std()
    data['Z200'] = (data['Z200'] - data['Z200'].mean()) / data['Z200'].std()
    data['ZBOT'] = (data['ZBOT'] - data['ZBOT'].mean()) / data['ZBOT'].std()

    data[list(set(data.columns) - {'Label'})] = \
        data[list(set(data.columns) - {'Label'})].astype('float32')
    data['Label'] = data['Label'].astype('int8')
    data = data[list(set(data.columns) - {'Label'}) + ['Label']]

    train = data[data['Label'] != -1]
    if oversampling:
        # FIXME: need a double check because of imbalanced duplicates
        train0 = train[train['Label'] == 0]
        train1 = train[train['Label'] == 1]
        train1 = train1.sample(frac=len(train0) / len(train1), replace=True)
        train2 = train[train['Label'] == 2]
        train2 = train2.sample(frac=len(train0) / len(train2), replace=True)
        train = pd.concat([train0, train1, train2]).sample(frac=1)\
            .reset_index(drop=True)

    test = data[data['Label'] == -1]
    test = test.drop(columns=['Label'])
    test.index -= 10000000

    return train, test


class LogisticRegression:
    """
    Logistic Regression model.
    """

    def __init__(
            self, in_dimension: int = 24,
            out_dimension: int = 3,
            learning_rate: float = 0.01,
            regularization: float = 0.1,
            initial_weights: np.ndarray = None,
            initial_bias: np.ndarray = None):
        """
        Initialize the model.

        """

        if out_dimension < 1 or in_dimension < 1:
            raise ValueError("in_dimension and out_dimension must be positive")

        # initializing weights
        if not isinstance(initial_weights, np.ndarray):
            self._weights = np.random.uniform(
                -1, 1, (in_dimension, out_dimension))
        elif initial_weights.shape == (in_dimension, out_dimension):
            self._weights = initial_weights
        else:
            raise ValueError("initial_weights has wrong shape")

        # initializing bias
        if not isinstance(initial_bias, np.ndarray):
            self._bias = np.random.uniform(-1, 1, (out_dimension,))
        elif initial_bias.shape == (out_dimension,):
            self._bias = initial_bias
        else:
            raise ValueError("initial_bias has wrong shape")

        # initializing other parameters
        self.learning_rate = learning_rate
        self.regularization = regularization

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
        gradient = data_x.T @ (predict_y - data_y) / len(data_x) + \
            self.regularization * self._weights
        gradient_bias = np.sum(predict_y - data_y) / len(data_x) + \
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
        return -(np.log(predict) * data_y).sum() /\
            (data_y.shape[0] * data_y.shape[1])

    @staticmethod
    def _precision(predict: np.ndarray, data_y: np.ndarray) -> np.ndarray:
        # This function calculates the precision
        # FIXME: It's not accurate
        return (predict.T @ data_y) / data_y.sum(0)

    @staticmethod
    def _accuracy(predict: np.ndarray, data_y: np.ndarray) -> float:
        # This function calculates the accuracy
        return (predict.argmax(1) == data_y.argmax(1)).sum() / len(data_y)

    @staticmethod
    def _fit_epoch_list(show: bool, epochs: int) -> list:
        # This function returns the list of epochs to show in fit method
        return list({0}.union(
            i for i in range(epochs) if (i + 1) % (epochs // 20) == 0
            ).union({epochs - 1})) if show else {}

    def fit(self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            print_loss: bool = False,
            validate_x: np.ndarray = None,
            validate_y: np.ndarray = None,
            epochs: int = 100) -> (list, list, list, list):
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
        validate_x : np.ndarray
            Validation data.
            shape: (m, in_dimension)
        validate_y : np.ndarray
            Validation labels.
            shape: (m, out_dimension) or (m,)
        epochs : int
            Number of epochs to train the model.

        Returns
        -------
        Tuple[list, list, list, list]
            Train loss, validation loss, train accuracy, validation accuracy.
            over all epochs.

        Raises
        ------
        ValueError
            If the input data has wrong shape.
            train_x and validate_x must have shape (n, in_dimension).
            train_y and validate_y must have shape (n, out_dimension).
        """

        train_x = self._process_data_x(train_x)
        train_y = self._process_data_y(train_y)
        train_loss = [self._loss(self._predict(train_x), train_y)]
        train_accuracy = [self._accuracy(self._predict(train_x), train_y)]

        validate_loss = validate_accuracy = None
        if validate_x is not None and validate_y is not None:
            validate_x = self._process_data_x(validate_x)
            validate_y = self._process_data_y(validate_y)
            validate_loss = [self._loss(self._predict(validate_x), validate_y)]
            validate_accuracy =\
                [self._accuracy(self._predict(validate_x), validate_y)]

        print_epochs = self._fit_epoch_list(print_loss, epochs)

        for epoch in range(epochs):

            # FIXME: need a double check and adding more optimizers
            predict = self._predict(train_x)
            gradient, gradient_bias = self._gradient(train_x, predict, train_y)

            self._weights -= self.learning_rate * gradient
            self._bias -= self.learning_rate * gradient_bias

            train_loss.append(self._loss(predict, train_y))
            train_accuracy.append(self._accuracy(predict, train_y))
            if validate_x is not None and validate_y is not None:
                valid_predict = self._predict(validate_x)
                validate_loss.append(self._loss(valid_predict, validate_y))
                validate_accuracy.append(
                    self._accuracy(valid_predict, validate_y))

            if validate_loss is not None and epoch in print_epochs:
                print(
                    f"Epoch {epoch + 1}:\n"
                    f"Train loss: {train_loss[-1]}\n"
                    f"Train accuracy: {train_accuracy[-1]}\n"
                    f"Validation loss: {validate_loss[-1]}\n"
                    f"Validation accuracy: {validate_accuracy[-1]}\n")
            elif epoch in print_epochs:
                print(
                    f"Epoch {epoch + 1}:\n"
                    f"Train loss: {train_loss[-1]}\n"
                    f"Train accuracy: {train_accuracy[-1]}\n")

        return (train_loss, validate_loss, train_accuracy, validate_accuracy)

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

        return self._predict(self._process_data_x(data_x))

    def validate(
            self,
            validate_x: np.ndarray,
            validate_y: np.ndarray) -> (np.ndarray, np.ndarray):
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
        Tuple[np.ndarray, np.ndarray]
            accuracy, precision
        """

        validate_x = self._process_data_x(validate_x)
        validate_y = self._process_data_y(validate_y)

        predict = self._predict(validate_x)
        return (
            self._accuracy(predict, validate_y),
            self._precision(predict, validate_y)
        )


def main(args):
    """
    Main function.
    """

    train, test = pre_process(
        args.train_data,
        args.test_data,
        not args.not_oversampling
    )

    train_x = np.array(
        train[list(set(train.columns) - {'Label'})],
        dtype='float32'
    )
    train_y = np.array(train['Label'], dtype='int8')
    train_y = (train_y == np.unique(train_y)[:, None]).T

    model = LogisticRegression(
        train_x.shape[1],
        train_y.shape[1],
        args.learning_rate,
        args.regularization
    )

    model.fit(train_x, train_y, args.print_loss, epochs=args.epochs)
    test['Label'] = model.predict(test).argmax(1).astype('int8')
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
    parser.add_argument('train_data', type=str, help='Path to the train data')
    parser.add_argument('test_data', type=str, help='Path to the test data')
    parser.add_argument('-o', '--output', type=str, default='output.csv',
                        help='Path to the output file')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('-rg', '--regularization', type=float, default=0.1,
                        help='Regularization')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('-p', '--print_loss', action='store_true',
                        help='Print loss and accuracy during training')
    parser.add_argument('--not_oversampling', action='store_false',
                        help='Do not over-sample the train data')
    main(parser.parse_args())