import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from losses import kl_divergence, reconstruction_loss

print(tf.__version__)

# defince a sampling layer
class Sampling(layers.Layer):
    #Uses (z_mean, z_var) to sample z
    def call(self, z_mean, z_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

class vae(keras.Model):
    def __init__(self, latent_dim = 2, **kwargs):
        super(vae, self).__init__(**kwargs)
        # encoder
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_var = layers.Dense(latent_dim, name='z_var')(x)
        z = Sampling()(z_mean, z_var)
        encoder = keras.Model(encoder_inputs, [z_mean, z_var, z], name = 'encoder')
        
        # decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
            ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss_val = reconstruction_loss(data, reconstruction)
            kl_loss = kl_divergence(z_mean, z_var)
            total_loss = reconstruction_loss_val + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss_val)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
            }

# vae = vae(latent_dim = 2)
# vae.compile(optimizer=keras.optimizers.Adam())
# vae.fit(mnist_digits, epochs= 3, batch_size=128)        
        



