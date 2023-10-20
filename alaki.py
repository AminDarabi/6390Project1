from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


def read_climate_data(data_path='data/train.csv') -> pd.DataFrame:

    data = pd.read_csv(data_path, index_col=0)

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

    # modifying wind data
    data['U850inv'] = data['U850'].apply(lambda x: x if x < 0 else 0)
    data['U850'] = data['U850'].apply(lambda x: x if x > 0 else 0)

    data['V850inv'] = data['V850'].apply(lambda x: x if x < 0 else 0)
    data['V850'] = data['V850'].apply(lambda x: x if x > 0 else 0)

    data['UBOTinv'] = data['UBOT'].apply(lambda x: x if x < 0 else 0)
    data['UBOT'] = data['UBOT'].apply(lambda x: x if x > 0 else 0)

    data['VBOTinv'] = data['VBOT'].apply(lambda x: x if x < 0 else 0)
    data['VBOT'] = data['VBOT'].apply(lambda x: x if x > 0 else 0)

    # data = data.drop(columns=['lat', 'lon'])

    return data


train = read_climate_data('data/train.csv')
test = read_climate_data('data/test.csv').astype('float32')

train_x = train.drop('Label', axis=1).astype('float32')
train_y = train['Label'].astype('int8')

train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.08)
valid1_x, valid2_x, valid1_y, valid2_y = train_test_split(
    valid_x, valid_y, test_size=0.5)

transform = MaxAbsScaler().fit(train_x)
train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)

model = MLPClassifier(
    hidden_layer_sizes=(100, 20, 10), solver='lbfgs', verbose=True,
    max_iter=50000).fit(transform.transform(train_x), train_y)

print(model.score(transform.transform(train_x), train_y))
print(model.score(transform.transform(valid_x), valid_y))
print(model.score(transform.transform(valid1_x), valid1_y))
print(model.score(transform.transform(valid2_x), valid2_y))

test['Label'] = model.predict(transform.transform(test))
test['Label'].to_csv(
    f'result/output_{date.today()}_100_100.csv', header=True)
