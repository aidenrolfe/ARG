import matplotlib.pyplot as plt
import numpy as np 
import random

## reading galaxy images and redshifts 
gal_input = np.load('inputgalaxies.npy')
z_in = np.load('inputredshifts.npy')    
gal_target = np.load('targetgalaxies.npy')
z_out = np.load('targetredshifts.npy')
convolved_image = np.load('convolvedimage.npy')
target_convolved_image = np.load('convolvedtargetimage.npy')

m = gal_input.shape[0]
n = random.randint(0,m-1) # choosing a random galaxy from sample to plot

## plot random input galaxy
for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(gal_input[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Redshift z = ' + str(np.round(z_in[n],2)) + ' with galaxy image ' + str(n+1))
plt.show()

## plot random target galaxy
for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(gal_target[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Redshift z = ' + str(np.round(z_out[n],2)) + ' with galaxy image ' + str(n+1))
plt.show()

## plot input after PSF
for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(convolved_image[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Redshift z = ' + str(np.round(z_in[n],2)) + ' with galaxy image ' + str(n+1) + ' after PSF applied')
plt.show()

## plot target after PSF
for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(target_convolved_image[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Redshift z = ' + str(np.round(z_out[n],2)) + ' with galaxy image ' + str(n+1) + ' after PSF applied')
plt.show()
