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
m = random.randint(0,n-1)

image = gal_input[m,...] # testing with one input image (all filters)

norm = image / np.max(image)
scaled_image = norm * 1000

# observing simulated galaxy:
# resize object to fit kernel
def observe_gal(image, input_redshift, output_redshift, seeing):
    image = resize(image, input_redshift, output_redshift)
    return image

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


# changes to brightness (probably not keeping)
def dimming(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift)
    d_o = cosmo.luminosity_distance(output_redshift)
    dimming = (d_i / d_o)**2
    
    dimmed = np.empty((17, 60, 60))
    for a in range(17):
        dimmed[a] = image[...,a] * dimming
    return dimmed
    
dimmed_image = dimming(image, input_redshift, output_redshift)

for b in range(17):
    plt.subplot(3,6,b+1)
    plt.imshow(image[...,b], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before dimming') 
plt.show()  

for c in range(17):
    plt.subplot(3,6,c+1)
    plt.imshow(dimmed_image[c], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after dimmming') 
plt.show()


convolved_image, target_convolved_image = convolve_psf(image,seeing)


# Can't get the below to work for a test across all filters yet - 
dimmed = dimming(convolved_image, input_redshift, output_redshift)

# saving convolved input and target image
np.save('convolvedimage.npy', convolved_image)
np.save('convolvedtargetimage.npy', target_convolved_image)


# changes to size:
def rebinning(image, input_redshift, output_redshift):
    d_i = (cosmo.luminosity_distance(input_redshift))
    d_o = (cosmo.luminosity_distance(output_redshift))
    scale_factor = (d_i.value / (1 + input_redshift)**2) / (d_o.value / (1 + output_redshift)**2)
    rebinned = np.empty((17, 60, 60))
    for r in range(17):
        rebinned[r] = zoom(image[...,r], scale_factor)
    return rebinned

# rebinned image:
rebinned_image = rebinning(image, input_redshift, output_redshift)
 
for d in range(17):
    plt.subplot(3,6,d+1)
    plt.imshow(image[...,d], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before rebinning') 
plt.show() 

for e in range(17):
    plt.subplot(3,6,e+1)
    plt.imshow(rebinned_image[e], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after rebinning') 
plt.show() 
    
# does this need to be normalised by dividing by the sum of the image/flux?
   
    
    
# adding shot noise (from variations in the detection of photons from the source):
def add_shot_noise(scaled_image):    
    with_shot_noise = np.empty((17, 60, 60))
    for s in range(17):
        with_shot_noise[s] = np.random.poisson(scaled_image[...,s])
    return with_shot_noise

# image with noise:
image_with_shot_noise = add_shot_noise(scaled_image)

for f in range(17):
    plt.subplot(3,6,f+1)
    plt.imshow(image[...,f], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before shot noise') 
plt.show() 

for g in range(17):
    plt.subplot(3,6,g+1)
    plt.imshow(image_with_shot_noise[g], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after shot noise') 
plt.show() 



# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(scaled_image):
    # peak needs to be tested to find a suitable value to determine the amount of background noise added to be realistic
    with_background = np.empty((17, 60, 60))
    for x in range(17):
        with_background[x] = np.random.normal(0.0, 3,scaled_image[...,x])
    return with_background

# image with background:
image_with_background = add_background(scaled_image)

for y in range(17):
    plt.subplot(3,6,y+1)
    plt.imshow(image[...,y], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before background') 
plt.show() 

for z in range(17):
    plt.subplot(3,6,z+1)
    plt.imshow(image_with_background[z], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after background') 
plt.show() 




# saving data for use in VAE:
def save_data(image):
    # code to save data for VAE - need to edit
    return image
