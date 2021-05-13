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

# conditional = input('Conditional? [yes or no] ')

# digit = int(input('What digit? '))

conditional = 'yes'
digit = 3

# create digit filter
train_filter = y_train == digit
test_filter =y_test == digit

# apply filter to MNIST data set
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

labels = y_test

#x_train = x_train[y_train == 5]
#x_test = x_test[y_test == 5]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# find array shapes
n_train, w_train, h_train, c_train = x_train.shape
n_test, w_test, h_test, c_test = x_test.shape

c_train = 3 # set equal to 3 so imshow will produce a colour image
c_test = 3

x_train_colour = x_train * np.ones(c_train)
x_test_colour = x_test * np.ones(c_test)

# gives the r, g, b level for each of the n images

colours_train = np.random.uniform(size=(n_train, c_train))
colours_test = np.random.uniform(size=(n_test, c_test))

# add in the missing width and height axes

colours_train = colours_train[:, None, None, :]
colours_test = colours_test[:, None, None, :]

x_train_colour = x_train * colours_train
x_test_colour = x_test * colours_test

noise_factor = 0.5
x_train_noisy = x_train_colour + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_colour.shape) 
x_test_noisy = x_test_colour + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_colour.shape) 

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

latent_dim = 5

encoder_inputs = keras.Input(shape=(28, 28, 3))
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
if conditional=='yes':
    x = layers.Concatenate()([x, condition_inputs])
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name="encoder")
else:
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()    

# build the decoder

latent_inputs = keras.Input(shape=(latent_dim,))
# We add our conditions again here. One might think this is
# unnecessary, as the latent encoding already contains this
# information. However, I think including them here has two
# advantages: (1) it means that the latent encoding does not need to
# contain the condition, so can be more efficient, and (2) if we use
# the decoder alone, we can specify the condition.
if conditional=='yes':
    x = layers.Concatenate()([latent_inputs, condition_inputs])
    x = layers.Dense(7 * 7 * 64, activation="relu")(x)
else:
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
if conditional=='yes':
    decoder = keras.Model([latent_inputs, condition_inputs], decoder_outputs, name="decoder")
else:
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# define the VAE model

def reconstruction_loss(targets, outputs):
    loss = keras.losses.mean_squared_error(outputs, targets)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2)))
    return loss

beta = 0.1
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

if conditional=='yes':
    vae_outputs = decoder([encoder([encoder_inputs, condition_inputs])[-1],
                           condition_inputs])
    vae = keras.Model([encoder_inputs, condition_inputs], vae_outputs)
else:
    vae_outputs = decoder(encoder(encoder_inputs)[-1])
    vae = keras.Model(encoder_inputs, vae_outputs)
vae.add_loss(kl_loss)
vae.add_metric(kl_loss, name='kl_loss')

vae.compile(optimizer=keras.optimizers.Adam(), loss=reconstruction_loss,
            metrics=[reconstruction_loss])

# train the VAE

# This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

epochs = 10
batch_size = 128

logdir = "/tmp/tb/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

if conditional=='yes':
    history = vae.fit([x_train_noisy, y_train], x_train_colour,
                      epochs=epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      validation_data=([x_test_noisy, y_test], x_test_colour),
                      callbacks=[tensorboard_callback, early_stopping_callback])
    
    reconstructions = vae.predict([x_test_noisy, y_test])
    z_mean, z_log_var, z = encoder.predict([x_test_noisy, y_test])
else:
    history = vae.fit(x_train_noisy, x_train_colour,
                      epochs=epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      validation_data=(x_test_noisy, x_test_colour),
                      callbacks=[tensorboard_callback, early_stopping_callback])
    
    reconstructions = vae.predict(x_test_noisy)
    z_mean, z_log_var, z = encoder.predict(x_test_noisy)

# summarize history for loss

plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.savefig('model_loss.pdf')

# show what the original, noisy and reconstructed digits look like

n = 10
fig, axarr = plt.subplots(3, n, figsize=(20, 6))
for i, ax in enumerate(axarr[0]):
    ax.imshow(x_test_colour[i], cmap='gray')
for i, ax in enumerate(axarr[1]):
    ax.imshow(x_test_noisy[i], cmap='gray')
for i, ax in enumerate(axarr[2]):
    ax.imshow(reconstructions[i], cmap='gray')
for ax in axarr.flat:
    ax.axis('off')
plt.savefig('examples.pdf')

# display a 2D plot of the digit classes in the latent space

fig, axarr = plt.subplots(figsize=(6, 6))
plt.scatter(z[:, 0], z[:, 1], c=labels, marker='.')
plt.axis('square')
plt.colorbar()
plt.title('Digit Classes in Latent Space')
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.savefig('latent_scatter.pdf')
