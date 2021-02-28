import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime

# Creates noisy digits

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#x_train = x_train[y_train == 5]
#x_test = x_test[y_test == 5]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Our conditions will be the labels of the input digits.
# We could perhaps just give the label itself here (divided
# by num_classes in order to normalise):
#y_train = y_train.astype('float32') / num_classes
#y_test = y_test.astype('float32') / num_classes
# However, these are categories, rather than a continuum,
# so in this case it is probably more appropriate to use a set
# of num_classes inputs, giving the probability that the label
# belongs to each class (these will all be zero, except for one,
# which will be one - a so-called "one hot" encoding).
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# create a sampling layer

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# build the encoder

latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
# We need another input for the labels.
# If our condition were a continuous quantity, this could just be
# that value, but here we have a set of categorical classes:
condition_inputs = keras.Input(shape=(num_classes,))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
# I suggest we try including the conditions here. This means they can
# be processed (by a couple of fully-connected layers and one
# non-linearity) in producing the latent encoding, but avoids some
# complications of adding them in earlier. This is ok, because the
# basic features that the network needs to learn should not depend
# very strongly on the labels, but including them here should allow
# our encoding to be more efficient representation of each class.
x = layers.Concatenate()([x, condition_inputs])
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name="encoder")
encoder.summary()    

# build the decoder

latent_inputs = keras.Input(shape=(latent_dim,))
# We add our conditions again here. One might think this is
# unnecessary, as the latent encoding already contains this
# information. However, I think including them here has two
# advantages: (1) it means that the latent encoding does not need to
# contain the condition, so can be more efficient, and (2) if we use
# the decoder alone, we can specify the condition.
x = layers.Concatenate()([latent_inputs, condition_inputs])
x = layers.Dense(7 * 7 * 64, activation="relu")(x)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model([latent_inputs, condition_inputs], decoder_outputs, name="decoder")
decoder.summary()

# define the VAE model

def reconstruction_loss(targets, outputs):
    loss = keras.losses.mean_squared_error(outputs, targets)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2)))
    return loss

beta = 1.0
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

vae_outputs = decoder([encoder([encoder_inputs, condition_inputs])[-1],
                       condition_inputs])
vae = keras.Model([encoder_inputs, condition_inputs], vae_outputs)
vae.add_loss(kl_loss)
vae.add_metric(kl_loss, name='kl_loss')

vae.compile(optimizer=keras.optimizers.Adam(), loss=reconstruction_loss,
            metrics=[reconstruction_loss])

# train the VAE

logdir = "/tmp/tb/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

vae.fit([x_train_noisy, y_train], x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=([x_test_noisy, y_test], x_test),
        callbacks=[tensorboard_callback])

# show what the original, noisy and reconstructed digits look like

reconstructions = vae.predict([x_test_noisy, y_test])

n = 10
fig, axarr = plt.subplots(3, n, figsize=(20, 6))
for i, ax in enumerate(axarr[0]):
    ax.imshow(x_test[i], cmap='gray')
for i, ax in enumerate(axarr[1]):
    ax.imshow(x_test_noisy[i], cmap='gray')
for i, ax in enumerate(axarr[2]):
    ax.imshow(reconstructions[i], cmap='gray')
for ax in axarr.flat:
    ax.axis('off')
plt.savefig('examples.pdf')
