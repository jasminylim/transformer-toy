# File: compare.py
# Description: compare LSTM and Transformer performance
# Author: Jasmin Lim
# Date Created: June 2, 2023

import numpy as np
from scipy.integrate import solve_ivp

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
    ax[0].plot(data[:800,0], data[:800,2], '-b', label=r"Data")
    ax[0].set_xlabel(r"Time [s]")
    ax[0].set_ylabel(r"$u_2$")
    ax[0].set_title(r"Training Data")

    ax[1].plot(data[800:,0], data[800:,2], '-b', label=r"Data")
    filenames = ["data/lstm_e0.1_B2.0.csv", "data/tf_e0.1_B2.0.csv"]
    labels = [r"LSTM", r"Transformer"]

    for (i,file) in enumerate(filenames):
        data = np.loadtxt(file, delimiter=",")
        ax[1].plot(data[:,0], data[:,1], '--', label=labels[i])

    ax[1].set_xlabel(r"Time [s]")
    ax[1].set_ylabel(r"$u_2$")
    ax[1].set_ylim([-2,3])
    ax[1].legend()
    ax[1].set_title(r"Validation Data")

    plt.savefig("compare.png", dpi=400)

    
