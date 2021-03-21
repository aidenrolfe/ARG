import numpy as np
from scipy.ndimage import zoom
from astropy.convolution import convolve_fft
import astropy.convolution as conv
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# read in galaxy: 
image = np.load('inputgalaxies.npy')
input_redshift = np.load('inputredshifts.npy')    
target_image = np.load('targetgalaxies.npy')
output_redshift = np.load('targetredshifts.npy')

n = image.shape[0]

sd = 1.5 # std dev of gaussian
seeing = 2.354*sd # FWHM ~ 2.354*sd


# convolving image with gaussian psf
def convolve_psf(image, seeing):
    psf = conv.Gaussian2DKernel(seeing, x_size=60, y_size=60)
    convolved = np.empty((100, 60, 60, 17))
    target_convolved = np.empty((100, 60, 60, 17))
    # convolve PSF with images in all 17 filters
    for h in range(n):
        for i in range(17):
            convolved[h,...,i] = convolve_fft(image[h,...,i], psf.array)
            target_convolved[h,...,i] = convolve_fft(target_image[h,...,i], psf.array)
    return convolved, target_convolved

convolved_image, target_convolved_image = convolve_psf(image,seeing)

# changes to brightness (probably not keeping)
def dimming(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift[0])
    d_o = cosmo.luminosity_distance(output_redshift[0])
    dimming = (d_i / d_o)
    dimmed = convolved_image*dimming
    return dimmed

dimmed = dimming(convolved_image, input_redshift, output_redshift)

# saving convolved input and target image
np.save('convolvedimage.npy', convolved_image)
np.save('convolvedtargetimage.npy', target_convolved_image)


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
    with_shot_noise = np.random.poisson(image)
    return with_shot_noise

# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(image):
    # code to generate background noise, background = ...
    with_background = np.random.normal(...)
    return with_background

# saving data for use in VAE:
def save_data(image):
    # code to save data for VAE - need to edit
    return image