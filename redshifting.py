import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from astropy import convolution as conv
from astropy import cosmology as cosmo
import cv2


# read in galaxy:
with open('inputgalaxies.npy','rb') as f:
    gal_input = np.load(f)
    input_redshift = np.load(f)
        
with open('targetgalaxies.npy','rb') as g:
    gal_target = np.load(g)
    output_redshift = np.load(g)

image = gal_input[0,...] # testing with first input image

image = cv2.resize(image,(81,81)) # resize image to fit kernel

# # observing simulated galaxy:
# def observe_gal(image, input_redshift, output_redshift, seeing):
#     image = resize(image, input_redshift, output_redshift)
#     return image

seeing = 10 # just for testing

# convolving with a psf:
def convolve_psf(image, seeing):
    psf = conv.Gaussian2DKernel(seeing)
    convolved = np.convolve(image[...,0], psf)
    return convolved

convolved_image = convolve_psf(image[...,0],seeing)

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
def add_shot_noise(image, output_exptime):            
    # shot_noise = np.sqrt(convolved * output_exptime) * np.random.poisson()
    with_shot_noise = image + shot_noise
    return with_shot_noise

# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(image):
    # code to generate background noise, background = ...
    with_background = image + background
    return with_background

# saving data for use in VAE:
def save_data(image):
    # code to save data for VAE - need to edit
    return image