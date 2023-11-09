# IFT6390 Kaggle Competition 1: Climate Classification

This project contains two methods for the first IFT6390 Kaggle competition,
which aims to classify the weather situation based on time, location, and
other features.

- The first method is a logistic regression model with L2 regularization
developed from scratch.
- The second method is a multi-layer perceptron model developed using
SciKit-Learn.

Requirements
Both methods are implemented in Python 3.11 and require the following packages:
- NumPy
- Pandas

The second method also requires:
- SciKit-Learn

### Installation
To install the project, you can clone this repository or download it as a zip
file. Then, you can install the required packages using pip:
```
$ pip install -r requirements.txt
```

### Usage
To run the models and generate predictions for the Kaggle competition,
you can use one of the following files, based on the method you want to use.

## base.py
This file contains the logistic regression model with L2 regularization
developed from scratch. It has the following functions and classes:

- read_climate_data: This function reads the climate data from the csv file and
returns a Pandas DataFrame. It also applies some preprocessing on the data.
- LogisticRegression: This class implements the logistic regression model with
L2 regularization. It has the following public methods:
    - fit: This method trains the model on the given data and labels.
    - predict: This method predicts the labels for the given data.
    - validate: This method calculates the accuracy and loss of the model for
    the given data and labels.
- main: This function runs the model and prints the results using the arguments
provided by the user.

To run the model with default hyperparameters using the competition data,
you can use the following command in the terminal:
```
$ python base.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

You can also customize the following arguments:

- -rg or --reg: This argument specifies the regularization strength.
The default value is 0.2. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA -rg 0.1
    ```

- -ep or --epochs: This argument specifies the number of iterations for the
solver. The default value is 400. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA -ep 1000
    ```

- --solver: This argument specifies the solver to use for the optimization.
The available options are dummy and GradientDescent.
The default option is dummy. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA --solver GradientDescent
    ```

- --scaler: This argument specifies the scaler transformation to apply on the
data. The available options are AbsMax and MinMax.
The default option is AbsMax. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA --scaler MinMax
    ```

- --validate: This argument enables the validation mode, which splits the
training data into train and validation sets and evaluates the model on the
validation set before training on the whole training data. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA --validate
    ```

- --hyperparameter_tuning or -hps: This argument enables the hyperparameter
tuning mode, which uses a pre-defined policy to choose the best regularization
strength and number of epochs, and uses the best model to predict the labels
for the test data. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA --hyperparameter_tuning
    ```

- --print: This argument enables the print mode, which prints some brief
information including the loss and accuracy during training and
hyperparameter tuning. For example:
    ```
    $ python base.py TRAIN_DATA TEST_DATA --print
    ```

- -h or --help: This argument shows a help message with the description and
usage of the arguments. For example:
    ```
    $ python base.py -h
    ```

## advanced.py
This file contains the multi-layer perceptron model developed using
SciKit-Learn. It has the following functions:

- read_climate_data: This function reads the climate data from the csv file and
returns a Pandas DataFrame. It also applies some preprocessing on the data.
- main: This function runs the model and prints the results using the arguments
provided by the user.

To run the model with default hyperparameters using the competition data,
you can use the following command in the terminal:
```
$ python advanced.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

However, you can also customize the following arguments:

- -ep or --epochs: This argument specifies the maximum number of iterations for
the solver. The default value is 1000. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA -ep 200
    ```

- --alpha: This argument specifies the regularization strength.
The default value is 0.01. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --alpha 0.001
    ```

- --solver: This argument specifies the solver to use for the optimization.
The available options are lbfgs, sgd, and adam.
The default option is adam. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --solver sgd
    ```

- --activation: This argument specifies the activation function to use for the
hidden layers. The available options are identity, logistic, tanh, and relu.
The default option is relu. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --activation tanh
    ```

- --validate: This argument enables the validation mode, which splits the
training data into train and validation sets and evaluates the model on the
validation set before training on the whole training data. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --validate
    ```

- --layers: This argument specifies the size of the hidden layers.
The default value is (4, 2). You have to provide in form of a tuple.
like (100, 50, 10). For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --layers (4, )
    ```

- --hyperparameter_tuning or -hps: This argument enables the hyperparameter
tuning mode, which uses a pre-defined policy to choose the best regularization
strength and hidden layer size, and uses the best model to predict the labels
for the test data. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA -hps
    ```

- --print: This argument enables the print mode, which prints some brief
information during training and hyperparameter tuning. For example:
    ```
    $ python advanced.py TRAIN_DATA TEST_DATA --print
    ```

- -h or --help: This argument shows a help message with the description and
usage of the arguments. For example:
    ```
    $ python advanced.py -h
    ```

## Acknowledgments
This project, developed by Amin Darabi, is based on the first Kaggle
competition of the IFT6390 course at the University of Montreal.
The data and the problem description can be found on the [Kaggle page](https://www.kaggle.com/competitions/classification-of-extreme-weather-events-udem).
The code is mainly inspired by the course's lectures and tutorials, with some
additional references.
References, algorithm details, and methodologies are found in the report.