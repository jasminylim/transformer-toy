# File: compare.py
# Description: compare LSTM and Transformer performance
# Author: Jasmin Lim
# Date Created: June 2, 2023

import numpy as np

# Plotting
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

    data = np.loadtxt("data/vdp_e0.1_B2.0.csv", delimiter=",")

    fig, ax = plt.subplots(2, 1, figsize=(25,15))
    # ax[0].plot(data[:800,0], data[:800,2], '-b', label=r"Data")
    # ax[0].set_xlabel(r"Time [s]")
    # ax[0].set_ylabel(r"$u_2$")
    # ax[0].set_title(r"Training Data")
    # reconstruction data
    ax[0].plot(data[800:,0], data[800:,2], '-k', linewidth=2.2, label=r"Data")
    filenames = ["data/lstm_e0.1_B2.0_recon.csv", "data/tf_e0.1_B2.0_recon.csv"]
    labels = [r"LSTM", r"Transformer"]
    colors = ["--b", "--r"]

    for (i,file) in enumerate(filenames):
        nn_data = np.loadtxt(file, delimiter=",")
        ax[0].plot(nn_data[:,0], nn_data[:,1], colors[i], label=labels[i], linewidth=2.5)

    ax[0].set_xlabel(r"Time [s]")
    ax[0].set_ylabel(r"$u_2$")
    ax[0].set_ylim([-2,3])
    ax[0].legend()
    ax[0].set_title(r"Reconstruction")

    # prediction data
    ax[1].plot(data[800:,0], data[800:,2], '-k', label=r"Data", linewidth=2.2)
    filenames = ["data/lstm_e0.1_B2.0_pred.csv", "data/tf_e0.1_B2.0_pred.csv"]
    labels = [r"LSTM", r"Transformer"]

    for (i,file) in enumerate(filenames):
        nn_data = np.loadtxt(file, delimiter=",")
        ax[1].plot(nn_data[:,0], nn_data[:,1], colors[i], label=labels[i], linewidth=2)

    nn_data = np.loadtxt("data/arima_e0.1_B2.0_pred_u1.csv", delimiter=",")
    ax[1].plot(nn_data[:,0], nn_data[:,1], "--g", linewidth=2, label="ARIMA")
    ax[1].set_xlabel(r"Time [s]")
    ax[1].set_ylabel(r"$u_2$")
    ax[1].set_ylim([-2,3])
    ax[1].legend()
    ax[1].set_title(r"Prediction")

    plt.savefig("Figures/compare.png", dpi=400)

    
