import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import manifold
import random
import argparse
import os

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            # Turn on memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
        # Restrict TensorFlow to only use one of the GPUs
        tf.config.set_visible_devices(random.choice(gpus), "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Using GPUs: {logical_gpus}")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument("runname", type=str)
parser.add_argument("input", type=str, default="inputgalaxies_obs_nonoise")
parser.add_argument("target", type=str, default="targetgalaxies_obs_nonoise")
parser.add_argument("--unconditional", dest="conditional", action="store_false")
args = parser.parse_args()

# read in galaxy data
images_file = f"{args.input}.npy"
redshifts_file = f"{args.input.split('_')[0].replace('galaxies', 'redshifts')}.npy"
gal_input = np.load(images_file)
z_input = np.load(redshifts_file)

images_file = f"{args.target}.npy"
redshifts_file = f"{args.target.split('_')[0].replace('galaxies', 'redshifts')}.npy"
gal_target = np.load(images_file)
z_target = np.load(redshifts_file)

redshifts = np.transpose([z_input, z_target]) # combine redshifts into 1 array

# shuffle and then split galaxy and redshift data into test and train sets
gal_input_train, gal_input_test, gal_target_train, gal_target_test, redshifts_train, redshifts_test \
    = train_test_split(gal_input, gal_target, redshifts, test_size=0.2, shuffle=True)
    
# normalize each exampld
def norm(x):
    r = x.reshape((x.shape[0], -1))
    scale = r.std(axis=-1)
    r = r / scale[:, None]
    x = r.reshape(x.shape)
    return scale, x

gal_input_train_scale, gal_input_train = norm(gal_input_train)
gal_input_test_scale, gal_input_test = norm(gal_input_test)
gal_target_train_scale, gal_target_train = norm(gal_target_train)
gal_target_test_scale, gal_target_test = norm(gal_target_test)

n_input_train, w, h, c = gal_input_train.shape
n_input_test, _, _, _ = gal_input_test.shape

z_condition = 2

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

latent_dim = 20

encoder_inputs = keras.Input(shape=(w, h, c))
condition_inputs = keras.Input(shape=(z_condition,))
x = layers.GaussianNoise(stddev=0.2)(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", strides=3, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=3, padding="same")(x)
x = layers.Flatten()(x)

if args.conditional:
    x = layers.Concatenate()([x, condition_inputs])
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name="encoder")
else:
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()    

# build the decoder

latent_inputs = keras.Input(shape=(latent_dim,))
if args.conditional:
    x = layers.Concatenate()([latent_inputs, condition_inputs])
    x = layers.Dense(7 * 7 * 64, activation="relu")(x)
else:
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="valid")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(c, 3, padding="same")(x)
if args.conditional:
    decoder = keras.Model([latent_inputs, condition_inputs], decoder_outputs, name="decoder")
else:
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# define the VAE model

def reconstruction_loss(targets, outputs):
    loss = keras.losses.mean_squared_error(outputs, targets)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2)))
    return loss

beta = 1
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

if args.conditional:
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

def plot_example(input, target, reconstructions, residuals, redshifts, filename="examples.pdf"):
    m = target.shape[0]
    r = random.randint(0, m-1) # choosing a random galaxy to plot (as input and target redshift)
    vmax = target[r].max()
    fig, axarr = plt.subplots(5, c, figsize=(c*2, 8))
    for i, ax in enumerate(axarr[0]):
        ax.imshow(input[r,:,:,i], cmap='inferno',
                   origin='lower', interpolation='nearest',
                   vmin=0, vmax=vmax)
        if i==0:
            ax.set_ylabel('input')
    for i, ax in enumerate(axarr[1]):
        ax.imshow(target[r,:,:,i], cmap='inferno',
                   origin='lower', interpolation='nearest',
                   vmin=0, vmax=vmax)
        if i==0:
            ax.set_ylabel('target')
    for i, ax in enumerate(axarr[2]):
        ax.imshow(reconstructions[r,:,:,i], cmap='inferno',
                   origin='lower', interpolation='nearest',
                   vmin=0, vmax=vmax)
        if i==0:
            ax.set_ylabel('reconstruction')  
    for i, ax in enumerate(axarr[3]):
        ax.imshow(residuals[r,:,:,i], cmap='coolwarm', 
                  origin='lower', interpolation='nearest',
                  vmin=-0.1*vmax, vmax=0.1*vmax)
        if i==0:
            ax.set_ylabel('residual (x10)') 
    for i, ax in enumerate(axarr[4]):
        ax_target = plt.subplot(5, 1, 5)
        ax_target.plot(np.arange(c), target[r].sum(axis=(0, 1)), 'bo-', label='target')
        ax_target.set_xlabel('filter number')
        ax_target.set_ylabel('target flux', color = 'b')
        ax_target.tick_params(axis='y', labelcolor='b') 
        ax_target.set_xlim(-0.5, c-0.5)
        ax_recon = ax_target.twinx()       
        ax_recon.plot(np.arange(c), reconstructions[r].sum(axis=(0, 1)), 'ro-', label='reconstructions')
        ax_recon.set_ylabel('reconstruction flux', color = 'r')
        ax_recon.tick_params(axis='y', labelcolor='r')
    for ax in axarr.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Galaxy image ' + str(r) + ' with input z = ' + str(np.round(redshifts[r,0],2)) \
                 + ' and target z = ' + str(np.round(redshifts[r,1],2)))
    plt.savefig(filename)
    
# Plotting projection of latent space with TSNE, with reconstructed images instead of points.      

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom, vae):
    """
    Inputs:
        x, y : TSNE points 
        imageData: Your images for testing 
    """
    images = []
    reconstructed_imgs = vae.predict(imageData)
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        for j in range(c):
            img = reconstructed_imgs[i,:,:,j]
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(x_test, z_test, encoder, vae, save=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    z_mean, z_log_var, z = encoder.predict([gal_input_test, redshifts_test])
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(z_mean) 

    # Plot images according to t-sne embedding
    if save:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=[x_test, z_test], ax=ax, zoom=0.3, vae = vae)
        plt.xlabel('z mean [0]')
        plt.ylabel('z mean [1]')
        plt.savefig('TSNE_scatter.png')
    else:
        return X_tsne

if __name__ == "__main__":

    # train the VAE
    
    # This callback will stop the training when there is no improvement in
    # the validation loss for 25 consecutive epochs
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    
    epochs = 1000
    batch_size = 128
    
    logdir = f"./runs/{args.runname}" 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    if args.conditional:
        history = vae.fit([gal_input_train, redshifts_train], gal_target_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          verbose=2,
                          validation_data=([gal_input_test, redshifts_test], gal_target_test),
                          callbacks=[tensorboard_callback, early_stopping_callback])
        
        reconstructions = vae.predict([gal_input_test, redshifts_test])
        z_mean, z_log_var, z = encoder.predict([gal_input_test, redshifts_test])
    else:
        history = vae.fit(gal_input_train, gal_target_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          verbose=2,
                          validation_data=(gal_input_test, gal_target_test),
                          callbacks=[tensorboard_callback, early_stopping_callback])
        
        reconstructions = vae.predict(gal_input_test)
        z_mean, z_log_var, z = encoder.predict(gal_input_test)
    
    vae.save_weights(os.path.join(logdir, 'weights')) # save model for future use
    
    # summarize history for loss
    
    plt.plot(history.history['loss'],'b')
    plt.plot(history.history['val_loss'],'r')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(logdir, 'model_loss.pdf'))
    
    # show what the original, simulated and reconstructed galaxies look like
    residuals = gal_target_test - reconstructions
    plot_example(gal_input_test, gal_target_test, reconstructions, residuals, redshifts_test,
                 filename=os.path.join(logdir, "examples.pdf"))
    
    # display a 2D plot of redshifting condition in the latent space
    
    # fig, axarr = plt.subplots(figsize=(6, 6))
    # plt.plot(z[:, 0], z[:, 1], 'k.')
    # plt.axis('square')
    # plt.title('Redshift Conditions in Latent Space')
    # plt.xlabel('z[0]')
    # plt.ylabel('z[1]')
    # plt.savefig(os.path.join(logdir, 'latent_scatter.pdf'))
          

    # display multi-dimensional latent plot
    
    computeTSNEProjectionOfLatentSpace(gal_input_test,redshifts_test,encoder,vae,save =True)
