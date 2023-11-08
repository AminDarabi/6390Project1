# 6390Project1

This Project contains two methods for first IFT6390 Kaggle competition.

- First method is a logistic regression model with L2 regularization developed
from scratch.
- Second method is a multi-layer perceptron model developed using scikit-learn.

Both methods are implemented in Python 3.11. and need the following packages:
- numpy
- pandas

second method also need:
- scikit-learn

## base.py

This file contains logistic regression model with L2 regularization developed
from scratch. It contains the following functions and classes:

- read_climate_data
    - This function reads the climate data from the csv file and returns
    pandas dataframe.
    - This function also apply some preprocessing on the data.
- LogisticRegression
    - This class implements logistic regression model with L2 regularization.
    - This class contains the following functions:
        - fit: This function trains the model.
        - predict: This function predicts the labels for the given data.
        - validate: This function calculates the accuracy and loss of the model for
        the given data.
- main
    - This function using arguments run the model and prints the results.

### How to run

To run the model with default hyper-parameters using competition data, you need
to run the following command in the terminal:
```
$ python base.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

You can also change regularization strength and maximum number of iterations
using -rg and -ep arguments respectively. For example:
```
$ python base.py TRAIN_DATA TEST_DATA -rg 0.1 -ep 1000
```

It is also possible to choose solver and scaler transformation using --solver
and --scaler arguments respectively. For example:
```
$ python base.py TRAIN_DATA TEST_DATA --solver GradientDescent --scaler AbsMax
```
dummy and GradientDescent are the only available solvers and AbsMax and
MinMax are the only available scalers.

If you want to validate the model using my validation policy, you can use
--validate argument. In this case program shows validation accuracy and loss
before training the model on the whole training data.
For example:
```
$ python base.py TRAIN_DATA TEST_DATA --validate
```

It is also possible to use a pre-defined policy to chose regularization
strength and maximum number of iterations, and use the best model to predict
the labels for the test data. To do so, you need to use --hyperparameter_tuning
or -hps argument. For example:
```
$ python base.py TRAIN_DATA TEST_DATA --hyperparameter_tuning
```

Using --print argument program prints some brief information during training
and hyper-parameter tuning. For example:
```
$ python base.py TRAIN_DATA TEST_DATA --print
```

For request a help message, you can use -h or --help argument. For example:
```
$ python base.py -h
```

## advanced.py

This file contains multi-layer perceptron model developed using scikit-learn.
It contains the following functions and classes:

- read_climate_data
    - This function reads the climate data from the csv file and returns
    pandas dataframe.
    - This function also apply some preprocessing on the data.
- main
    - This function using arguments run the model and prints the results.

### How to run

Very similar to base.py file. To run the model with default hyper-parameters
using competition data, you need to run the following command in the terminal:
```
$ python advanced.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

To specify the number of max iterations, and regularization strength, you can
use -ep and --alpha arguments respectively. For example:
```
$ python advanced.py TRAIN_DATA TEST_DATA -ep 1000 --alpha 0.001
```

For different solvers, and activation functions you can use --solver and
--activation arguments respectively. For example:
```
$ python advanced.py TRAIN_DATA TEST_DATA --solver lbfgs --activation tanh
```
only lbfgs, sgd, and adam are available solvers and identity, logistic, tanh,
and relu are available activation functions.

If you want to validate the model using my validation policy before starting
train on whole data and predict the labels for the test data, you can use
--validate argument. In this case program shows validation accuracy and loss
before training the model on the whole training data.
For example:
```
$ python advanced.py TRAIN_DATA TEST_DATA --validate
```

to specify size of hidden layers, you can use --layers argument. For
example:
```
$ python advanced.py TRAIN_DATA TEST_DATA --layers 100 50 10
```

It is also possible to use a pre-defined policy to chose regularization
strength and model, and use the best model to predict the labels for the test
data. To do so, you need to use --hyperparameter_tuning or -hps argument.
For example:
```
$ python advanced.py TRAIN_DATA TEST_DATA --hyperparameter_tuning
```

Using --print argument program prints some brief information during training
and hyper-parameter tuning. For example:
```
$ python advanced.py TRAIN_DATA TEST_DATA --print
```

For request a help message, you can use -h or --help argument. For example:
```
$ python advanced.py -h
```