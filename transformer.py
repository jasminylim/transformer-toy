# File: transformer.py
# Description: Train LSTM model
# Author: Jasmin Lim
# Date Created: June 1, 2023

import numpy as np
import pandas as pd

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import layers

# Import Time2Vec Layer
from time2vec import time2vec

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

class Transformer(tf.keras.Model):
    def __init__(self, input_size, head_size, num_heads, ff_dim, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.pos_encode = time2vec(input_shape[0])

    def transformer_encoder(self, inputs):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    
    def build(self):
        '''Build Transformer Model Architecture'''
        inputs = keras.Input(shape=self.input_size)
        x = self.pos_encode(inputs)
        x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        # for dim in self.mlp_units:
        #     x = layers.Dense(dim, activation="relu")(x)
        #     x = layers.Dropout(self.mlp_dropout)(x)

        outputs = layers.Dense(1)(x)

        return keras.Model(inputs, outputs)


if __name__=="__main__":
    ### --- DATA ---
    # import data
    # filename = "data/vdp_e0.1_B2.0.csv"
    filename = "data/square.csv"
    data = np.loadtxt(filename, delimiter=",")

    u = 1 # use x or dx/dt

    # split into training/validation series
    split = 100#800 # where to split training and validation data
    train_time, valid_time = data[:split,0], data[split:-8,0]
    train_data, valid_data = data[:split,u], data[split:-8,u]

    time_delay = 32
    batch_size = 16

    train_series = TimeseriesGenerator(train_data, train_data, length=time_delay, batch_size=batch_size)
    valid_series = TimeseriesGenerator(valid_data, valid_data, length=time_delay, batch_size=batch_size)

    ### --- TRAIN TRANSFORMER ---
    input_shape = (time_delay, 1)
    TF = Transformer(input_size=input_shape, head_size=32, num_heads=2, ff_dim=64, dropout=0)
    model = TF.build()

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-2))
    history = model.fit(train_series, validation_data=valid_series, steps_per_epoch=1, epochs=100, verbose=2)
    model.summary()

    ### --- PREDICTION ---
    pred_time = data[split:,0]
    nstep = len(pred_time)
    current = np.reshape(data[split-time_delay:split, u], (1, time_delay))
    pred_data = np.zeros(nstep)
    for n in range(nstep):
        pred_data[n] = model.predict(current, verbose=0)[0][0]
        current = np.hstack((current[0,1:], pred_data[n])).reshape((1, time_delay))
    # np.savetxt(f"data/tf_e{filename[10:13]}_B{filename[15:18]}_pred_u{u}.csv", np.transpose(np.vstack((pred_time, pred_data))), delimiter=",")
    np.savetxt(f"data/tf_square_pred.csv", np.transpose(np.vstack((pred_time, pred_data))), delimiter=",")

    x_pred, _ = delay_time_series(data[split-time_delay:,u], time_delay)
    pred_tf = model.predict(x_pred)
    # np.savetxt(f"data/tf_e{filename[10:13]}_B{filename[15:18]}_recon_u{u}.csv", np.transpose(np.vstack((pred_time, pred_tf.reshape((1,nstep))))), delimiter=",")
    np.savetxt(f"data/tf_square_recon.csv", np.transpose(np.vstack((pred_time, pred_tf.reshape((1,nstep))))), delimiter=",")

    fig, ax = plt.subplots(2,1,figsize=(15,10))
    ax[0].plot(train_time, train_data,"-b")
    ax[0].set_xlabel(r"t")
    # ax[0].set_ylabel(r"$u_2$")
    ax[0].set_title(r"Training Data")

    ax[1].plot(valid_time, valid_data, "-b", label="Data")
    ax[1].plot(pred_time, pred_data, "--r", label="Prediction")
    ax[1].plot(pred_time, pred_tf, "--g", label="Reconstruction")
    ax[1].set_xlabel(r"t")
    # ax[1].set_ylabel(r"$u_2$")
    ax[1].set_title(r"Validation Data")
    ax[1].legend(frameon=True)

    # plt.savefig(f"Figures/transformer_u{u}.png", dpi=400)
    plt.savefig("Figures/tf_square.png", dpi=400)

    # plot training/validation loss
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    ax.semilogy(history.history["loss"], '-k', label="Training Loss")
    ax.semilogy(history.history["val_loss"], '-r', label="Validation Loss")
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"MSE")
    ax.set_title(r"Transformer")
    ax.legend()

    # plt.savefig(f"Figures/tf_loss_u{u}.png", dpi=400)
    plt.savefig(f"Figures/tf_square_loss.png", dpi=400)
    


    
