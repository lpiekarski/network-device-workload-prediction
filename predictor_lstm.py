import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from math import nan

INPUT_FILE = "solution_template.csv"
print('reading data to predict')
data_to_predict = []
with open(INPUT_FILE) as f:
    lines = f.readlines()
    for line in lines:
        data_to_predict.append(line.split(",")[0:2])
DATE_ZERO = datetime.datetime.strptime('2019-12-02 08:00:00', '%Y-%m-%d %H:%M:%S')
#MAX_ID = 0
#MIN_ID = 1000000
SERIES_LENGTH = 1924
NANVAL = -1
#MIN_DATE = ""
#MAX_DATE = ""
#2020-02-20 12:00:00


def date_to_series_id(datestr):
    date = datetime.datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')
    delta = (date - DATE_ZERO)
    id = int(delta.total_seconds() / 3600.)
    #global MAX_ID, MAX_DATE, MIN_ID, MIN_DATE
    #if MAX_ID < id:
    #    MAX_ID = id
    #    MAX_DATE = datestr
    #if MIN_ID > id:
    #    MIN_ID = id
    #    MIN_DATE = datestr
    return id


def series_id_to_date(id):
    delta = datetime.timedelta(seconds=id * 3600.)
    datestr = datetime.datetime.strftime(DATE_ZERO + delta, '%Y-%m-%d %H:%M:%S')
    return datestr


def normalize(series):
    minval = np.nanmin(series)
    maxval = np.nanmax(series)
    for i in range(len(series)):
        if not np.isnan(series[i]):
            delta = maxval - minval
            if delta == 0:
                series[i] = 0
            else:
                series[i] = (series[i] - minval) / (maxval - minval)

    def ret(x):
        for i in range(len(x)):
            x[i] = x[i] * (maxval - minval) + minval
    return ret


def standardize(series):
    mean = np.nanmean(series)
    std = np.nanstd(series)
    for i in range(len(series)):
        if not np.isnan(series[i]):
            if std == 0:
                series[i] = 0
            else:
                series[i] = (series[i] - mean) / std

    def ret(x):
        for i in range(len(x)):
            x[i] = (x[i]) * std + mean
    return ret


print('reading and processing train data')
TRAIN_FILE = "training_series_long.csv"
hostname_series_dict = {}
hostname_series_reverse_dict = {}
read_lines = 0
train_data = []
for train_raw_data in pd.read_csv(TRAIN_FILE, engine='c', chunksize=1024):
    read_lines += 1024
    print(f"read lines: {read_lines}")
    for index, observation in train_raw_data.iterrows():
        key = (observation['hostname'], observation['series'])
        #print(f"{key} |||| {data_to_predict}")
        if key not in hostname_series_dict:
            if [observation['hostname'], observation['series']] not in data_to_predict:
                continue
            id = len(train_data)
            train_data.append(np.full((SERIES_LENGTH, 1), nan))
            hostname_series_dict[key] = id
            hostname_series_reverse_dict[id] = key
        train_data[hostname_series_dict[key]][date_to_series_id(observation['time_window']), 0] = float(observation['Mean'])
#for time_series in train_data:
#    plt.plot(time_series)
#    plt.show()
#print(MIN_ID)
#print(MAX_ID)
#print(MIN_DATE)
#print(MAX_DATE)
#for time_series in train_data:
#    print(time_series)
print("standarizing series")
reverse_transform = []
for time_series in train_data:
    reverse_transform.append(standardize(time_series))
    np.nan_to_num(time_series, nan=NANVAL, copy=False)
#for time_series in train_data:
#    print(time_series)
#    plt.plot(time_series)
#    plt.show()
train_data = np.asarray(train_data)
#print(train_data.shape)
data_X = [[] for _ in range(999)]
data_y = [[] for _ in range(999)]
CONSIDERED_VALUES = 32
WINDOW_SHIFT = 1
print('creating time windows for training')
ii = 0
for time_series in train_data:
    result_row = -1
    for j in range(len(data_to_predict)):
        if (data_to_predict[j][0], data_to_predict[j][1]) == hostname_series_reverse_dict[ii]:
            result_row = j
            break
    if result_row == -1:
        continue
    series_id = result_row
    print(f"series id: {series_id}")
    #data_X.append([])
    #data_y.append([])
    data_X[series_id] = np.empty((SERIES_LENGTH - CONSIDERED_VALUES, CONSIDERED_VALUES, 1))
    data_y[series_id] = np.empty((SERIES_LENGTH - CONSIDERED_VALUES, 1))
    for i in range(0, SERIES_LENGTH - CONSIDERED_VALUES, WINDOW_SHIFT):
        print(f"{i + 1}/{(SERIES_LENGTH - CONSIDERED_VALUES)}")
        data_X[series_id][i] = (tf.slice(time_series, [i, 0], [CONSIDERED_VALUES, 1]))
        data_y[series_id][i] = (tf.reshape(tf.slice(time_series, [i + CONSIDERED_VALUES, 0], [1, 1]), (1)))
    ii = ii + 1
    #data_X[series_id] = np.asarray(data_X[series_id])
    #data_y[series_id] = np.asarray(data_y[series_id])
print("reshaping")
data_X = np.asarray(data_X)
data_y = np.asarray(data_y)
#print(F"data shape: X:{data_X.shape}, y: {data_y.shape}")
#print(data_X[0][0])
#print(data_y[0][0])


def r_squared(y, y_pred):
    #print(f"y: {y}, y_pred: {y_pred}")
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1., tf.divide(residual, total))
    return r2


def r_squared_loss(y, y_pred):
    #print(f"y: {y}, y_pred: {y_pred}")
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2loss = tf.divide(residual, total)
    return r2loss


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=NANVAL),
        tf.keras.layers.LSTM(84),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=r_squared_loss, metrics=[r_squared, 'mae', 'mse'])
    return model


print(hostname_series_reverse_dict)
print(hostname_series_dict)
print(data_X.shape[0])
results = np.zeros((len(data_to_predict), 168))
for i in range(data_X.shape[0]):
    if not i in hostname_series_reverse_dict:
        break
    result_row = -1
    for j in range(len(data_to_predict)):
        if (data_to_predict[j][0], data_to_predict[j][1]) == hostname_series_reverse_dict[i]:
            result_row = j
            break
    if result_row == -1:
        continue
    print(f"training model for series {hostname_series_reverse_dict[i]}")
    model = create_model()
    #print(data_X[i][0].shape)
    #y_pred = model.predict(np.asarray([data_X[i][0]]))
    #print(f"pred: {y_pred[0]}, y: {data_y[i][0]}")
    X = data_X[i]#[~np.isnan(data_X[i])]
    y = data_y[i]#[~np.isnan(data_X[i])]
    model.fit(x=X, y=y, batch_size=128, validation_split=0.01, epochs=84)
    predicted = []
    train = train_data[i]#[~np.isnan(train_data[i])]
    for j in range(0, 168):
        #print(tf.slice(train_data[i], [SERIES_LENGTH - CONSIDERED_VALUES + j, 0], [CONSIDERED_VALUES - j, 1]).shape)
        #print(np.asarray(predicted).reshape(j, 1).shape)
        if j > CONSIDERED_VALUES:
            data = tf.slice(np.asarray(predicted).reshape((j, 1)), [j - CONSIDERED_VALUES, 0], [CONSIDERED_VALUES, 1])
        else:
            q = tf.slice(train, [SERIES_LENGTH - CONSIDERED_VALUES + j, 0], [CONSIDERED_VALUES - j, 1])
            data = tf.concat([q, np.asarray(predicted).reshape((j, 1))], 0)
        #print(np.asarray(data_X[i][data_X.shape[1] - 1]).shape)
        predicted.append(model.predict(np.asarray([data])))
    #plt.plot(tf.concat([train_data[i], np.asarray(predicted).reshape((168, 1))], 0))
    results[result_row] = reverse_transform[i](np.asarray(predicted).reshape((168, 1)))
    #plt.plot([x for x in range(0, train.shape[0])], train, 'r')
    #plt.plot([x for x in range(train.shape[0], train.shape[0] + 168)], np.asarray(predicted).reshape((168, 1)), 'b')
    #plt.show()
pd.DataFrame(results).to_csv('results.csv')