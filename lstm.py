# File: lstm.py
# Description: Train LSTM model
# Author: Jasmin Lim
# Date Created: May 31, 2023

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras_tuner as kerastuner

class LSTM_model(tf.keras.Model):
    def __init__(self, input_size, num_hidden):
        '''Build LSTM model
        Inputs:
            input_size [tuple] : NN input size
            num_hidden [int] : number of hidden layers in LSTM'''
        
        super().__init__()
        self.input_size = input_size
        self.num_hidden = num_hidden

    def build(self):
        '''Build LSTM Model Architecture'''

        input = keras.Input(shape=self.input_size)
        x = LSTM(self.num_hidden, input_shape=self.input_size)(input)
        output = Dense(1)(x)

        return keras.Model(input, output)

def delay_time_series(data, time_delay):
    '''delay_time_series : Create time series dataset with delay
    Inputs:
        data [array]: data array
        time_delay [int] : time_delay
    '''
    y = data[time_delay:]
    x = np.zeros((y.shape[0], time_delay))
    for i in range(x.shape[0]):
        x[i] = data[i:i+time_delay]

    return x, y

if __name__=="__main__":
    ### --- IMPORT TRAINING DATA ---
    filename = "data/vdp_e0.1_B2.0.csv"
    data = np.loadtxt(filename, delimiter=",")

    split = 800 # where to split training and validation data
    train_time, valid_time = data[:split,0], data[split:-8,0]
    train_data, valid_data = data[:split,2], data[split:-8,2]

    time_delay = 32
    batch_size = 16

    train_series = TimeseriesGenerator(train_data, train_data, length=time_delay, batch_size=batch_size)
    valid_series = TimeseriesGenerator(valid_data, valid_data, length=time_delay, batch_size=batch_size)

    # model = Sequential()
    # model.add(LSTM(32, input_shape=(time_delay, 1)))
    # model.add(Dense(1))
    LSTM_M = LSTM_model(input_size=(time_delay,1), num_hidden=32)
    model = LSTM_M.build()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='mse')
    model.fit(train_series, validation_data=valid_series, steps_per_epoch=1, epochs=100, verbose=2)
    model.summary()

    pred_time, pred_data = data[split:,0], data[split-time_delay:,2]
    x_pred, _ = delay_time_series(data[split-time_delay:,2], time_delay)
    pred_lstm = model.predict(x_pred)
    np.savetxt(f"data/lstm_e{filename[10:13]}_B{filename[15:18]}.csv", np.transpose(np.vstack((pred_time, pred_lstm[:,0]))), delimiter=",")

    fig, ax = plt.subplots(2,1,figsize=(15,10))
    ax[0].plot(train_time, train_data,"-b")
    # ax[0].plot(train_time[time_delay:], pred_lstm, "--r", label="LSTM Prediction")
    ax[0].set_xlabel(r"t")
    ax[0].set_ylabel(r"$u_2$")
    ax[0].set_title(r"Training Data")

    ax[1].plot(valid_time, valid_data, "-b", label="Data")
    ax[1].plot(pred_time, pred_lstm, "--r", label="LSTM Prediction")
    ax[1].set_xlabel(r"t")
    ax[1].set_ylabel(r"$u_2$")
    ax[1].set_title(r"Validation Data")
    ax[1].legend()

    plt.savefig("lstm.png", dpi=400)
    


    
