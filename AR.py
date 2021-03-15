import sys
import argparse
import numpy as np
import glob

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM  
from astropy.convolution import convolve
from astropy import constants as const
from astropy import units as u

from scipy.ndimage import zoom

from matplotlib import pyplot as plt



# default configuration in the event of missing data - is this needed?

class Config(object):
    '''
        Tracks all the flags used by Artificial Redshift.
        If none is provided, it loads the default defined below.
    '''

    #h = 0.7
    #cosmo = FlatLambdaCDM(H0=100 * h, Om0=0.3, Tcmb0=2.725)
    add_background = True
    rebinning = True
    convolve_with_psf = True
    make_cutout = True
    dimming = True
    shot_noise = True
    output_size = 128

    
# defining an 'observation' frame (for inital and target redshift, with the following parameters...)
    
class ObservationFrame(object):
    '''
        Class that represents one observation frame with a given
        instrument setup.
    '''

    def __init__(self, redshift, pixelscale, exptime):
        self.pixelscale = pixelscale
        self.redshift = redshift
        self.exptime = exptime      

        
# artificial redshifting (broken into separate stages)
        
class ArtificialRedshift(object):
    '''
        This handles all transformations and effects selected
        in the Config class to be applied to the input data,
        from initial_frame to target_frame. It keeps track
        of the transformation in the input image, it is possible 
        to retrieve partial results between each step, ideal for
        debugging.
    '''

    def __init__(self, image, psf, background, initial_frame, target_frame, MAG, config=None):

        self.image = image
        self.psf = psf
        self.background = background
        self.initial_frame = initial_frame
        self.target_frame = target_frame
        self.MAG = MAG

        if config is None:
            self.config = Config()

        self.geometric_rebinning() 
        self.apply_dimming()
        self.convolve_psf()
        self.apply_shot_noise()
        self.add_background()

    @classmethod
    def fromrawdata(cls, image,
                         psf,
                         background,
                         initial_redshift,
                         target_redshift, 
                         initial_pixelscale, 
                         target_pixelscale,
                         obs_exptime, 
                         target_exptime):

        current_frame = ObservationFrame(initial_redshift, initial_pixelscale, obs_exptime)
        target_frame = ObservationFrame(target_redshift, target_pixelscale, target_exptime)
    
        return cls(image, psf, background, initial_frame, target_frame)

        self.final = self.image

    
    # convolving with a psf: 
    
    def convolve_psf(self):
        
        if self.config.convolve_with_psf:
            
            original_flux = self.final.sum()

            self.psf /= self.psf.sum()
            self.convolved = convolve(self.final, self.psf)   
            self.final = self.convolved.copy()
        
        
    # changes to size: (uses luminosity_distance, but should angular_diameter_distance be used instead?)
            
    def geometric_rebinning(self):
        
        if self.config.rebinning:
           
            self.flux = self.final.sum()
            self.rebinned = self.final / self.flux

            initial_distance = self.cosmo.luminosity_distance(self.initial_frame.redshift).value   
            target_distance = self.cosmo.luminosity_distance(self.target_frame.redshift).value   
            self.scale_factor = (initial_distance * (1 + self.target_frame.redshift)**2) / (target_distance * (1 + self.initial_frame.redshift)**2)      # FERENGI - eq.1
            
            self.rebinned = zoom(self.rebinned, self.scale_factor, order=0, prefilter=True)
            self.rebinned /= self.rebinned.sum()        # this divides rebinned by rebinned.sum() and IMMEDIATELY applies it to rebinned again. i.e b /= a is also b = b/a
            self.rebinned *= self.flux      # alike to the above line
            
            self.final = self.rebinned.copy()
       
    
    # changes to brightness:           
            
    def apply_dimming(self):

        if self.config.dimming:
            self.dimming_factor = (self.cosmo.luminosity_distance(self.initial_frame.redshift) / self.cosmo.luminosity_distance(self.target_frame.redshift))**2     # FERENGI - eq.2
            self.dimming_factor = self.dimming_factor.value
            self.dimmed = self.final * self.dimming_factor
            self.final = self.dimmed.copy()

            
    # applying shot noise to the images (from variations in the detection of photons from the source):  
      
    def apply_shot_noise(self):         
        
        if self.config.shot_noise:
            self.shot_noise = np.sqrt(abs(self.convolved * self.target_frame.exptime)) * np.random.poisson(lam=self.target_frame.exptime, size=self.convolved.shape) / self.target_frame.exptime         
            self.with_shot_noise = self.final + self.shot_noise
            self.final = self.with_shot_noise.copy()

            
    # applying background noise to the images (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise)
    
    def add_background(self):

        if self.config.add_background:
           
            # edit this
        
            self.with_background = self.background.copy()
            self.with_background[offset_min:offset_max, offset_min:offset_max] += self.with_shot_noise
            self.final = self.with_background.copy()

            
    # writing data output for use in VAE:        
            
    def writeto(self, filepath, data, overwrite=False):

        hdr = fits.Header()

        if self.config.rebinning:
            hdr['REBIN'] = self.scale_factor
        
        if self.config.dimming:
            hdr['DIM_FACT'] = self.dimming_factor


        hdr['HISTORY'] = 'Image simulated with AREIA: Artificial Redshift Effects for IA'
        hdr['HISTORY'] = 'Source Extracted with Galclean'
        hdr['HISTORY'] = f'From z = {self.initial_frame.redshift} to z = {self.target_frame.redshift}'
        hdr['HISTORY'] = 'Any issues forward it to leonardo.ferreira@nottingham.ac.uk'
        hdr['HISTORY'] = 'Or seek help at https://github.com/astroferreira/areia'

        fits.writeto(filename=filepath, data=data, header=hdr, overwrite=overwrite)
