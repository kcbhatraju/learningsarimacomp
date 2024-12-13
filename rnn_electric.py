import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tqdm import tqdm

import tensorflow as tf

power_data = pd.read_csv("household_power_consumption.txt", sep=";", header=0, low_memory=False, na_values="?", infer_datetime_format=True, parse_dates={"datetime": [0, 1]}, index_col="datetime")

# plot global active power
global_active_power = power_data["Global_active_power"]

for i in range(len(global_active_power)):
    if pd.isna(global_active_power.iloc[i]):
        if i-24*60 < 0:
            global_active_power.iloc[i] = global_active_power.iloc[i+24*60]
        else:
            global_active_power.iloc[i] = global_active_power.iloc[i-24*60]

# average 24 * 60 observations --> 1 observation
global_active_power_daily = global_active_power.resample("D").mean()
global_active_power_daily = global_active_power_daily.values

plt.plot(global_active_power_daily)
plt.show()

print(len(global_active_power_daily)) # 1441
num_total = len(global_active_power_daily) # 1441

# idx 0 --> 1440
# last 7 are test data
# 1434 --> 1440 are test data
# 0 --> 1433 are train data

# in particular, we train by predicting idxs 365 --> 1433
# we test by our predictions on idxs 1434 --> 1440

# to predict t[i] for i in [365, ..., 1433]
# features are t[i-365], t[i-14], t[i-7], t[i-6], t[i-5], t[i-4], t[i-3], t[i-2], t[i-1]
# i.e. last year, two weeks ago, and all of last week

num_test_forecasts = 365

# i.e. for num_test_forecasts = 5
# test set consists of following predictions
# 1434 --> 1440
# 1433 --> 1439
# 1432 --> 1438
# 1431 --> 1437
# 1430 --> 1436
# so train set consists of 0 --> 1429
out_steps = 7
len_train = num_total-out_steps-num_test_forecasts+1

# we will use expanding window cv
# same model will be trained singular batches iteratively

# in particular, our goal is to forecast in batches of 7
# so to predict 365 -> 365+7
# we use the above features with i=365

# thus, i starts at 365, goes up by 1 each time, and ends at 1433-6=1427


input_datas = []
output_datas = []

rolling_means = []
rolling_stds = []

def collect_features(idx):
    rolling_mean = global_active_power_daily[:idx].mean()
    rolling_std = global_active_power_daily[:idx].std()

    just_last_year = global_active_power_daily[idx-365:idx-365+7].reshape(-1, 1)
    all_last_week = global_active_power_daily[idx-8:idx].reshape(-1, 1)
    input_data = np.concatenate([just_last_year, all_last_week], axis=0)

    output_data = global_active_power_daily[idx:idx+out_steps]

    input_data = (input_data - rolling_mean) / rolling_std
    output_data = (output_data - rolling_mean) / rolling_std

    return input_data, output_data, rolling_mean, rolling_std

num_val_batches = 10

# in the example with 0 --> 1429,
# we want i to range from 365 (where we would be predicting 365 -> 365+6 = 371)
# to 1423 (where we would be predicting 1423 -> 1423+6 = 1429)
for i in range(365, len_train-(num_val_batches+1)*out_steps+1):
    input_data, output_data, rolling_mean, rolling_std = collect_features(i)

    input_datas.append(input_data)
    output_datas.append(output_data)

    rolling_means.append(rolling_mean)
    rolling_stds.append(rolling_std)

input_datas = np.array(input_datas)
output_datas = np.array(output_datas)

print(input_datas.shape, output_datas.shape)

val_input_datas = []
val_output_datas = []

for i in range(len_train-(num_val_batches+1)*out_steps+1, len_train-out_steps+1):
    input_data, output_data, rolling_mean, rolling_std = collect_features(i)

    val_input_datas.append(input_data)
    val_output_datas.append(output_data)

val_input_datas = np.array(val_input_datas)
val_output_datas = np.array(val_output_datas)

print(val_input_datas.shape, val_output_datas.shape)

"""
# (1062, 9, 1) (1062, 7)
# CV procedure is as follows
# 10 fold cross validation
# last fold is training on batches 10 -> 1054, testing on batch 1055
# second last fold is training on batches 9 -> 1053, testing on batch 1054
# third last fold is training on batches 8 -> 1052, testing on batch 1053
# ...
# first fold is training on batches 1 -> 1045, testing on batch 1046

cv_mses = []
for cv_fold in tqdm(range(10)):
    train_data = (input_datas[cv_fold:1055-10+cv_fold], output_datas[cv_fold:1055-10+cv_fold])
    val_data = (input_datas[1055-10+cv_fold].reshape(1, 9, 1), output_datas[1055-10+cv_fold].reshape(1, 7))

    model = Feedback(32, out_steps)
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_data[0], train_data[1], epochs=10, batch_size=32)

    val_pred = model.predict(val_data[0])
    val_pred = val_pred * rolling_stds[1055-10+cv_fold] + rolling_means[1055-10+cv_fold]
    val_actual = val_data[1] * rolling_stds[1055-10+cv_fold] + rolling_means[1055-10+cv_fold]

    mse = np.mean((val_pred - val_actual)**2)
    cv_mses.append(mse)

plt.plot(cv_mses)
plt.show()"""

unit_sizes = [8, 16, 32, 64, 128]

validation_mses = []
validation_maes = []
validation_rmses = []

models = []
import time

before_cv = time.time()
for unit_size in unit_sizes:
    # now, we train on all data (1062 batches)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(unit_size, activation="relu"),
        tf.keras.layers.Dense(out_steps),
        tf.keras.layers.Reshape((out_steps, 1))
    ])

    # early stopping (patience=10)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(input_datas, output_datas, epochs=100, batch_size=32, validation_data=(val_input_datas, val_output_datas), callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

    mse = history.history["val_loss"][-1]
    mae = history.history["val_mae"][-1]
    rmse = history.history["val_root_mean_squared_error"][-1]

    validation_mses.append(mse)
    validation_maes.append(mae)
    validation_rmses.append(rmse)

    models.append(model)

after_cv = time.time()

print(f"CV Time: {(after_cv-before_cv)/len(unit_sizes)} sec/model")

print(validation_mses)
print(validation_maes)
print(validation_rmses)

idx = input("Which model to use? ")

plt.plot(history.history['loss'])
plt.show()

model = models[int(idx)]

test_input_datas = []
test_output_datas = []

test_forecasts = []
test_mses = []
for i in range(num_test_forecasts):
    idx = num_total-out_steps-num_test_forecasts+1+i
    # we are predicting idx -> idx+6

    input_data, output_data, rolling_mean, rolling_std = collect_features(idx)
    
    """rolling_mean = global_active_power_daily[:idx].mean()
    rolling_std = global_active_power_daily[:idx].std()

    just_last_year = global_active_power_daily[idx-365].reshape(1, 1)
    just_two_weeks_ago = global_active_power_daily[idx-14].reshape(1, 1)
    all_last_week = global_active_power_daily[idx-7:idx].reshape(7, 1)

    test_input_data = np.concatenate([just_last_year, just_two_weeks_ago, all_last_week], axis=0)
    test_input_data = (test_input_data - rolling_mean) / rolling_std
    test_input_data = test_input_data.reshape(1, 9, 1)
    test_input_data = tf.convert_to_tensor(test_input_data, dtype=tf.float32)"""
    test_input_datas.append(input_data)
    test_output_datas.append(output_data)

before_test = time.time()
test_input_datas = np.array(test_input_datas)
test_output_datas = np.array(test_output_datas)
test_preds = model.predict(test_input_datas)
after_test = time.time()

print(f"Test Time: {(after_test-before_test)} sec/model")

for i in range(num_test_forecasts):
    test_pred = test_preds[i]
    test_pred = test_pred * rolling_std + rolling_mean
    output_data = test_output_datas[i] * rolling_std + rolling_mean

    test_forecasts.append(test_pred.flatten())

    mse = np.mean((output_data - test_pred.flatten())**2)
    test_mses.append(mse)

    # plot forecst against actual
    """plt.plot(global_active_power_daily[idx:idx+7])
    plt.plot(test_forecasts[-1])
    plt.show()"""

print(np.mean(test_mses))
plt.plot(global_active_power_daily)
for i, forecast in enumerate(test_forecasts):
    # the ith forecast has len(train_data) + i points in front of it
    plt.plot([None]*(len_train+i) + list(forecast))
plt.title("Seven-Day Forecasts on Holdout Data (LSTM)")
plt.show()

# use averaging to get a single forecast
# each forecast is 7 days
# final_forecast[0] = test_forecasts[0][0]
# final_forecast[1] = (test_forecasts[0][1] + test_forecasts[1][0]) / 2
# final_forecast[2] = (test_forecasts[0][2] + test_forecasts[1][1] + test_forecasts[2][0]) / 3
# up to 6
# after 6
# final_forecast[i] = (test_forecasts[i-6][6] + test_forecasts[i-5][5] + ... + test_forecasts[i][0]) / 7

final_mse = 0
final_forecast = []
for i in range(num_test_forecasts):
    forecast = 0
    for j in range(min(out_steps, i+1)):
        forecast += test_forecasts[i-j][j]

    forecast /= min(out_steps, i+1)
    final_forecast.append(forecast)

num_before = len(global_active_power_daily) - num_test_forecasts
plt.plot(global_active_power_daily)
plt.plot([None]*num_before + final_forecast)
plt.title("Weighted-Average Forecasts on Holdout Data (LSTM)")
plt.show()

# print final mse
final_mse = np.mean((global_active_power_daily[num_before:] - final_forecast)**2)
print(final_mse)


"""
# predict test data
test_data = global_active_power_daily[1434:].values

# now rolling mean and std is just the mean and std of the entire train data
rolling_mean = train_data.mean()
rolling_std = train_data.std()

just_last_year = global_active_power_daily.values[1434-365].reshape(1, 1)
just_two_weeks_ago = global_active_power_daily.values[1434-14].reshape(1, 1)
all_last_week = global_active_power_daily.values[1434-7:1434].reshape(7, 1)
test_input_data = np.concatenate([just_last_year, just_two_weeks_ago, all_last_week], axis=0)

test_input_data = (test_input_data - rolling_mean) / rolling_std
test_input_data = test_input_data.reshape(1, 9, 1)

test_input_data = tf.convert_to_tensor(test_input_data, dtype=tf.float32)

test_pred = model.predict(test_input_data)
test_pred = (test_pred * rolling_std + rolling_mean).flatten()


plt.plot(test_data)
plt.plot(test_pred)
plt.show()

cv_window = 365

cv_mses = []
for i in range(len(train_data) - cv_window):
    input_data = train_data[i:i+cv_window]

    # train lstm on window_data, output is next 7 days
    input_data = input_data.values

    input_mean = input_data.mean()
    input_std = input_data.std()

    input_data = (input_data - input_mean) / input_std
    input_data = input_data.reshape(-1, 1)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

    output_data = train_data[i+cv_window:i+cv_window+7]
    output_data = output_data.values
    output_data = (output_data - input_mean) / input_std
    output_data = output_data.reshape(-1, 1)
    output_data = tf.convert_to_tensor(output_data, dtype=tf.float32)

    model = Feedback(128, 7)
    model.compile(optimizer="adam", loss="mse")
    model.fit(input_data, output_data, epochs=10)

    # now validate using output_data as input


plt.plot(cv_mses)
plt.show()"""
