
import tensorflow as tf
from tensorflow import keras


def kl_divergence(z_mean, z_var):
    kl_loss = -0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var))
    return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

def reconstruction_loss(real, reconstruction):
  return tf.reduce_mean(
      tf.reduce_sum(
          keras.losses.binary_crossentropy(real, reconstruction), axis=(1, 2)
      )
  )
