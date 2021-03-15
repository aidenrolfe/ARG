import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
<<<<<<< HEAD
from astropy import convolution as conv
from astropy import cosmology as cosmo
from scipy import signal as sig
import random


# read in galaxy:
with open('inputgalaxies.npy','rb') as f:
    gal_input = np.load(f)
    input_redshift = np.load(f)
        
with open('targetgalaxies.npy','rb') as g:
    gal_target = np.load(g)
    output_redshift = np.load(g)

n = gal_input.shape[0]
m = random.randint(0,n-1)

image = gal_input[m,...] # testing with one input image (all filters)

sd = 1.5 # std dev of gaussian
seeing = 2.354*sd # FWHM ~ 2.354*sd

# observing simulated galaxy:
# resize object to fit kernel
def observe_gal(image, input_redshift, output_redshift, seeing):
    image = resize(image, input_redshift, output_redshift)
    return image


# convolving image with gaussian psf:
def convolve_psf(image, seeing):
    psf = conv.Gaussian2DKernel(seeing, x_size=60, y_size=60)
    plt.imshow(psf.array, interpolation='none', origin='lower')
    convolved = np.empty((17, 60, 60))
    for i in range(17):
        convolved[i] = sig.convolve2d(image[...,i], psf.array, mode = 'same')
    return convolved

convolved_image = convolve_psf(image,seeing)

for j in range(17):
    plt.subplot(3,6,j+1)
    plt.imshow(image[...,j], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before PSF') 
plt.show  

for k in range(17):
    plt.subplot(3,6,k+1)
    plt.imshow(convolved_image[k], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after PSF') 
plt.show()   


# changes to brightness:
def dimming(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift)
    d_o = cosmo.luminsoity_distance(output_redshift)
    dimming = (d_i / d_o)**2
    dimmed = image * dimming
    return dimmed

# changes to size:
def rebinning(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift)
    d_o = cosmo.luminsoity_distance(output_redshift)
    scale_factor = (d_i / (1 + input_redshift)**2) / (d_o / (1 + output_redshift)**2)
    rebinned = zoom(image, scale_factor)
    return rebinned
=======
from astropy import convolution 
from astropy import cosmology as cosmo


# read in galaxy:
def read_gal(galaxy):

  read(galaxy) 

# observing simulated galaxy:
def observe_gal(image, input_redshift, output_redshift, seeing):

  image = resize(image, input_redshift, outputput_redshift)
  return image

# convolving with a psf:
def convolve_psf(image, seeing):
  
  psf = Gaussian2Dkernel(seeing)
  convolved = convolve(image, psf)
  return convoled

# changes to brightness:
def dimming(image, input_redshift, output_redshift):

  d_i = cosmo.luminosity_distance(input_redshift)
  d_o = cosmo.luminsoity_distance(output_redshift)
  dimming = (d_i / d_o)**2
  dimmed = image * dimming
  return dimmed

# changes to size:
def rebinning(image, input_redshift, output_redshift):

  d_i = cosmo.luminosity_distance(input_redshift)
  d_o = cosmo.luminsoity_distance(output_redshift)
  scale_factor = (d_i / (1 + input_redshift)**2) / (d_o / (1 + output_redshift)**2)
  rebinned = zoom(image, scale_factor)
  return rebinned
>>>>>>> 64c5c27d679515c2d48b82759c549e6a12661418
  
  # does this need to be normalised by dividing by the sum of the image/flux?
    
# adding shot noise (from variations in the detection of photons from the source):
<<<<<<< HEAD
def add_shot_noise(image, output_exptime):            
    # shot_noise = np.sqrt(convolved * output_exptime) * np.random.poisson()
    with_shot_noise = np.random.poisson(image)
    return with_shot_noise

# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(image):
    # code to generate background noise, background = ...
    with_background = np.random.normal(mean=0, peak, image)
    return with_background

# saving data for use in VAE:
def save_data(image):
    # code to save data for VAE - need to edit
    return image
=======
def add_shot_noise(image):         
    
  with_shot_noise = np.random.poisson(image)
  return with_shot_noise

# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(image):
  
  # peak needs to be tested to find a suitable value to determine the amount of background noise added to be realistic
  with_background = np.random.normal(mean=0, peak, image)
  return with_background

# saving data for use in VAE:
def save_data(image):

  # code to save data for VAE - need to edit
  return image
>>>>>>> 64c5c27d679515c2d48b82759c549e6a12661418
