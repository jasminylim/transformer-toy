# File: arima.py
# Description: Train ARIMA model
# Author: Jasmin Lim
# Date Created: June 7, 2023

import pmdarima as pm
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt
import matplotlib
import niceplots
niceplots.setRCParams()
matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': 'cmr10',
                            'mathtext.fontset': 'cm',
                            'axes.formatter.use_mathtext': True,
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            })

if __name__=="__main__":
    ### --- DATA ---
    # import data
    # filename = "data/vdp_e0.1_B2.0.csv"
    filename = "data/square.csv"
    data = np.loadtxt(filename, delimiter=",")

    u = 1 # use x or dx/dt

    # split into training/validation series
    split = 100#800 # where to split training and validation data
    train_time, valid_time = data[:split,0], data[split:,0]
    train_data, valid_data = data[:split,u], data[split:,u]

    # auto_arima = pm.auto_arima(train_data)
    # print(auto_arima)
    # print(auto_arima.summary())
    # assert 1==2

    # model = ARIMA(train_data, order=(10,0,4)) #u2
    # model = ARIMA(train_data, order=(12,0,2)) #u1
    # model = ARIMA(train_data, order=(2,0,2)) #sine
    # model = ARIMA(train_data, order=(2,0,1)) #dine
    model = ARIMA(train_data, order=(2,0,3)) #square
    model_fit = model.fit()
    print(model_fit.summary())
    
    fore = model_fit.forecast(401)
    recon = model_fit.predict(dynamic=False)

    # np.savetxt(f"data/arima_e{filename[10:13]}_B{filename[15:18]}_pred_u{u}.csv", np.transpose(np.vstack((valid_time, fore))), delimiter=",")
    np.savetxt(f"data/arima_square_pred.csv", np.transpose(np.vstack((valid_time, fore))), delimiter=",")

    fig, ax = plt.subplots(1,1,figsize=(15,8))
    ax.plot(data[:,0], data[:,u], "-k", label="Data")
    ax.plot(train_time, recon, "--b", label="Training Reconstruction")
    ax.plot(valid_time, fore, "--r", label="Prediction")

    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(r"$u_1$")
    ax.set_title(r"ARIMA")
    ax.set_ylim([-3, 3.5])
    ax.legend()

    plt.savefig("Figures/arimax_test.png", dpi=400)
