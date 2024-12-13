import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tqdm import tqdm

global_active_power = pd.read_csv("household_power_consumption.txt", sep=";", header=0, low_memory=False, na_values="?", infer_datetime_format=True, parse_dates={"datetime": [0, 1]}, index_col="datetime")["Global_active_power"]

# fill na value with value 24 hours ago if possible, color this red in plot
# other wise fill with observation right before it, color this green in plot
"""for i in range(len(global_active_power)):
    if pd.isna(global_active_power.iloc[i]):
        if i-24*60 < 0:
            global_active_power.iloc[i] = 
        else:
            global_active_power.iloc[i] = global_active_power.iloc[i-24*60]"""

colors = []
for i in range(len(global_active_power)):
    if pd.isna(global_active_power.iloc[i]):
        global_active_power.iloc[i] =global_active_power.iloc[i-24*60]
        colors.append("red")
    else: colors.append("blue")

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# average 24 * 60 observations --> 1 observation
global_active_power_daily = global_active_power.resample("D").mean()

# --------------------------------------------------------------------

# --------------------------------------------------------------------
num_total = len(global_active_power_daily.values) # 1441
num_test_forecasts = 365
out_steps = 7
len_train = num_total-out_steps-num_test_forecasts+1
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# for each batch of 1440 observations, color blue if no imputation at all
# color green if at least one imputation with observation right before it
# color red if at least one imputation with observation 24 hours ago
# red trumps green trumps blue
colors_daily = []
for i in range(len(global_active_power_daily)):
    curr_range = slice(i*24*60, (i+1)*24*60)
    if "red" in colors[curr_range]:
        colors_daily.append("red")
    else:
        colors_daily.append("blue")

# make size of scatters default size
plt.scatter(global_active_power_daily.index, global_active_power_daily, c=colors_daily, s=plt.rcParams['lines.markersize'] ** 2/2)
plt.axvline(x=global_active_power_daily.index[len_train], color="green")
plt.xlabel("Date")
plt.ylabel("Global Active Power (Daily)")
plt.title("Global Active Power (Daily) with Imputation")
plt.show()

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# plot all data
plt.plot(global_active_power_daily.values)
# draw red line at len_train
plt.axvline(x=len_train, color='r')
plt.title("Global Active Power (Daily)")
plt.show()

# plt acf and pacf
plot_acf(global_active_power_daily.values, lags=500, title="ACF of Global Active Power (Daily)")
plt.show()

plot_pacf(global_active_power_daily.values, lags=500, title="PACF of Global Active Power (Daily)")
plt.show()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# seasonal difference with period 365
global_active_power_daily_seasonal = global_active_power_daily.diff(365).dropna()

# plot acf and pacf
plot_acf(global_active_power_daily_seasonal.values, lags=500, title="ACF of Seasonally Differenced Global Active Power (Daily)")
plt.show()

plot_pacf(global_active_power_daily_seasonal.values, lags=500, title="PACF of Seasonally Differenced Global Active Power (Daily)")
plt.show()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
global_active_power_daily = global_active_power_daily.values

def sine_linear_regression(y_values):
    x_features = np.array([np.array([1, np.cos(2*np.pi*i/365), np.sin(2*np.pi*i/365)]) for i in range(len(y_values))])
    cov_matrix = np.linalg.inv(x_features.T @ x_features)

    return cov_matrix @ x_features.T @ y_values

sine_params = sine_linear_regression(global_active_power_daily[:len_train])

# plot sine fit
plt.plot(global_active_power_daily[:len_train])
plt.plot([sine_params[0] + sine_params[1]*np.cos(2*np.pi*i/365) + sine_params[2]*np.sin(2*np.pi*i/365) for i in range(len_train)])
plt.xlabel("Day")
plt.ylabel("Global Active Power (Daily)")
plt.title("Sinusoidal Fit")
plt.show()

residuals = global_active_power_daily - np.array([sine_params[0] + sine_params[1]*np.cos(2*np.pi*i/365) + sine_params[2]*np.sin(2*np.pi*i/365) for i in range(len(global_active_power_daily))])

plt.plot(residuals)
plt.xlabel("Day")
plt.ylabel("Residuals")
plt.title("Sinusoidal Fit Residuals")
plt.axvline(x=len_train, color='r')
plt.show()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# log transform (variance stabilizing)
# translate up by 2 to avoid log(negative)
res_log = np.log(residuals + 5.)
# --------------------------------------------------------------------
plt.plot(res_log)
plt.axvline(x=len_train, color='r')
plt.xlabel("Day")
plt.ylabel("Scaled Residuals")
plt.title("Variance-Stabilized Residuals")
plt.show()

# plot acf and pacf
plot_acf(res_log, lags=500, title="ACF of Variance-Stabilized Residuals")
plt.show()

plot_pacf(res_log, lags=500, title="PACF of Variance-Stabilized Residuals")
plt.show()

plot_pacf(res_log, lags=50, title="PACF of Variance-Stabilized Residuals (Zoomed In)")
plt.show()
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# cross validate AR model based on AIC and MSE
# cross-validate AR(7...14) models
import time

# track cv time
do_cv = True
if do_cv:
    ar_vals = list(range(7, 12))

    cv_aics = [[] for _ in ar_vals]
    cv_bics = [[] for _ in ar_vals]
    cv_mses = [[] for _ in ar_vals]

    before_cv = time.time()

    min_to_train = 1000
    for ar_val in tqdm(ar_vals):
        for i in tqdm(range(min_to_train, len_train-out_steps+1)):
            if (i != len_train-out_steps): continue
            
            # subtract mean
            res_log_mean = np.mean(res_log[:i])
            input_data = res_log[:i] - res_log_mean

            model = ARIMA(input_data, order=(ar_val, 0, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(out_steps)

            aic = model_fit.aic
            cv_aics[ar_val-7].append(aic)

            bic = model_fit.bic
            cv_bics[ar_val-7].append(bic)

            # add back mean
            forecast = forecast + res_log_mean

            # undo log transform
            forecast = np.exp(forecast) - 5

            # add back sine fit
            forecast = forecast + np.array([sine_params[0] + sine_params[1]*np.cos(2*np.pi*i/365) + sine_params[2]*np.sin(2*np.pi*i/365) for i in range(i, i+out_steps)])

            mse = np.mean((global_active_power_daily[i:i+out_steps] - forecast)**2)
            cv_mses[ar_val-7].append(mse)
    
    after_cv = time.time()

    print(f"CV Time: {(after_cv-before_cv) / len(ar_vals)} sec/model")


    print(f"AICs: {cv_aics}")
    print(f"BICs: {cv_bics}")
    print(f"MSEs: {cv_mses}")
    
    # plot cv metrics
    for i in range(len(ar_vals)):
        plt.plot(cv_aics[i], label=f"AR({ar_vals[i]})")
    plt.legend()
    plt.show()

    for i in range(len(ar_vals)):
        plt.plot(cv_bics[i], label=f"AR({ar_vals[i]})")
    plt.legend()
    plt.show()

    for i in range(len(ar_vals)):
        plt.plot(cv_mses[i], label=f"AR({ar_vals[i]})")

    plt.legend()
    plt.show()

# best model AR(8)
full_res_mean = np.mean(res_log[:len_train])
model = ARIMA(res_log - full_res_mean, order=(8, 0, 0))
model_fit = model.fit()

# get residuals of model
model_resid = model_fit.resid

# plot acf and pacf of residuals
plot_acf(model_resid, lags=500, title="Autocorrelation After Fit")
plt.show()

plot_pacf(model_resid, lags=500, title="Partial Autocorrelation After Fit")
plt.show()

test_forecasts = []
test_mses = []
before_test = time.time()
for i in range(num_test_forecasts):
    # i want to incorporate the test data into the model (i am only wanting to forecast 7 days at a time)
    idx = num_total-out_steps-num_test_forecasts+i
    # i want to apply model_fit to the data up to idx WITHOUT reestimating the model
    
    # subtract mean
    res_log_mean = np.mean(res_log[:idx])
    res_log_norm = res_log[:idx] - res_log_mean

    model_clone = model_fit.apply(res_log_norm)
    forecast = model_clone.forecast(out_steps)

    # add back mean
    forecast = forecast + res_log_mean

    # undo log transform
    forecast = np.exp(forecast) - 5

    # add back sine fit
    forecast = forecast + np.array([sine_params[0] + sine_params[1]*np.cos(2*np.pi*i/365) + sine_params[2]*np.sin(2*np.pi*i/365) for i in range(idx, idx+out_steps)])

    test_forecasts.append(forecast)

    mse = np.mean((global_active_power_daily[idx:idx+out_steps] - forecast)**2)
    test_mses.append(mse)
after_test = time.time()

print(f"Test Time: {(after_test-before_test)} sec/model")

plt.plot(global_active_power_daily)
for i in range(num_test_forecasts):
    plt.plot([None]*(len_train+i) + list(test_forecasts[i]))
plt.title("Seven-Day Forecasts on Holdout Data (SARIMA)")
plt.show()

# print average test mse
print(np.mean(test_mses))

# final forecast
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
plt.title("Weighted-Average Forecasts on Holdout Data (SARIMA)")
plt.show()

# print final mse
final_mse = np.mean((global_active_power_daily[num_before:] - final_forecast)**2)
print(final_mse)
