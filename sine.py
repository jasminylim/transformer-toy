# File: sine.py
# Description: generate sine data
# Author: Jasmin Lim
# Date Created: June 14, 2023

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
    # set simulation time
    t0 = 0; tf = 50
    nsteps = 500
    t = np.linspace(t0, tf, nsteps+1)

    ### --- sine wave ---
    sine = np.sin(0.5*np.pi*t)
    np.savetxt(f"data/sine.csv", np.transpose(np.vstack((t, sine))), delimiter=",")

    ### --- dissipating sine wave ---
    d_sine = np.exp(-t/2/np.pi)*np.cos(0.5*np.pi*t)
    np.savetxt(f"data/dsine.csv", np.transpose(np.vstack((t, d_sine))), delimiter=",")


    ### --- square wave ---
    square = np.zeros(len(sine))
    square[sine>0] = 1
    square[sine<0] = -1
    np.savetxt(f"data/square.csv", np.transpose(np.vstack((t, square))), delimiter=",")


    fig, ax = plt.subplots(1,3,figsize=(25,6))
    ax[0].plot(t, sine, "-k")
    ax[1].plot(t, d_sine, "-k")
    ax[2].plot(t, square, "-k")

    T = [r"Sine", r"Dissipative Sine", r"Square"]
    for i in range(3):
        ax[i].set_xlabel(r"$t$")
        ax[i].set_title(T[i])
        ax[i].set_ylim([-1.5, 1.5])
    plt.savefig("Figures/sine.png", dpi=400)