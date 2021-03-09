import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
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
  
  # does this need to be normalised by dividing by the sum of the image/flux?
    
# adding shot noise (from variations in the detection of photons from the source):
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
