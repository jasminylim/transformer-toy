# File: vdp.py
# Description: contains functions for Van der Pol's Equations
# Author: Jasmin Lim
# Date Created: May 30, 2023

import numpy as np
from scipy.integrate import solve_ivp

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

def vdp_lin(t, u, e=0.1, B=2.):
    '''vdp_lin : Van der Pol Equation
    Inputs:
        t [float] : time
        u [array] : state values
        e [float] : VDP constant
        B [float] : VDP constant'''
    dudt = np.zeros(u.shape)
    dudt[0] = u[1]
    dudt[1] = e*(1-u[0]**2)*u[1]-u[0]**3+B*np.cos(t)
    return dudt


if __name__=="__main__":
    ### --- RUN VAN DER POL SIMULATION ---
    e = 0.1
    B = 2.

    vdp = lambda t, u : vdp_lin(t, u, e, B)
    # set simulation time
    t0 = 0; tf = 200
    nsteps = 1000
    t = np.linspace(t0, tf, nsteps+1)

    # initializaion
    u0 = np.array([0., 0.])

    # solve IVP
    sol = solve_ivp(vdp, t_span=[t0,tf], y0=u0, t_eval=t, method="RK23")
    np.savetxt(f"data/vdp_e{e}_B{B}.csv", np.transpose(np.vstack((sol.t, sol.y))), delimiter=",")
    
    x = sol.y[0,:]
    dxdt = sol.y[1,:]
    d2xdt2 = e*(1-x**2)*dxdt-x**3+B*np.cos(t)

    ### --- PLOT RESULTS ---
    fig, ax = plt.subplots(3, 1, figsize=(10,17))

    ax[0].plot(x, dxdt, "-b")
    ax[0].set_xlabel(r"$u_1$")
    ax[0].set_ylabel(r"$u_2$")

    ax[1].plot(t, x, "-b", label=r"$u_1$")
    ax[1].plot(t, dxdt, "-r", label=r"$u_2$")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$\mathbf{u}$")
    ax[1].legend()

    ax[2].plot(x, d2xdt2, "-b")
    ax[2].set_xlabel(r"$x$")
    ax[2].set_ylabel(r"$d^2 x / dt^2$")

    plt.suptitle(r"$B$ = {}, $e$ = {}".format(B,e), fontsize=24)

    plt.savefig("Figures/vdp.png", dpi=400)

    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    start = 499
    ax.plot(t[start:start+32+1], dxdt[start:start+32+1], "-bo")
    ax.scatter(t[start+32+1], dxdt[start+32+1], color="red")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u_2$")
    plt.savefig("Figures/time_delay.png", dpi=400)






    

