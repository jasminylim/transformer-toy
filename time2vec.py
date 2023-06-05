import tensorflow as tf
from tensorflow.keras.layers import Layer

class time2vec(Layer):
    def __init__(self, time_delay):
        super().__init__()
        self.time_delay = time_delay

    def build(self, input_size):
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.time_delay),),
                                    initializer='uniform',
                                    trainable=True)
        
        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.time_delay),),
                                    initializer='uniform',
                                    trainable=True)
        
        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.time_delay),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.time_delay),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x, axis=-1) # Reduce dimension to (batch, time_delay)

        time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, time_delay, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic) # Periodic time feature
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, time_delay, 1)

        return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, time_delay, 2)


