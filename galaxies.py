import h5py
import numpy as np
from astropy.modeling.models import Sersic2D
import random

## HDF file produced by running candels_example.py
filename = "candels.goodss.models.test.hdf"
f = h5py.File(filename,'r')
list(f.keys())

z = np.array(f['z']) # redshift values
wl = np.array(f['wl']) # wavelength for each filter
flux = np.array(f['fluxes']) # flux at each redshift + filter

## reshaping flux array into (redshifts,filters,fluxes)
def flux_reshape(flux):    
    a = flux.shape[0]
    b = flux.shape[1]
    c = flux.shape[2]*flux.shape[3]*flux.shape[4]*flux.shape[5]*flux.shape[6]
    flux0 = np.reshape(flux,(a,b,c))
    return flux0

flux0 = flux_reshape(flux)

n = 100 # number of galaxies generated

## random selection of SEDs
def choose_seds(flux0):
    sed_idx = np.random.choice(flux0.shape[2], size=n) # array of n SEDs
    gal_seds = flux0[:,:,sed_idx]
    return gal_seds

gal_seds = choose_seds(flux0)
    
r = z.shape[0]

## random selection for input and target galaxy redshift    
def input_target(r,gal_seds):
    z_in_idx = random.randint(0,round((r-1)/2))
    z_out_idx = random.randint(1,round((r-1)/2)) + z_in_idx
    gal_seds_in = gal_seds[z_in_idx]*1e9
    gal_seds_out = gal_seds[z_out_idx]*1e9
    return gal_seds_in, z_in_idx, gal_seds_out, z_out_idx

gal_seds_in, z_in_idx, gal_seds_out, z_out_idx = input_target(r,gal_seds)

## generate sersic galaxies
np.random.seed(4251)
elip = np.round(np.random.uniform(low=0.0, high=0.8, size=(n,)), decimals=2)
PAs = np.round(np.random.uniform(low=0.0, high=np.pi, size=(n,)), decimals=2)
Reff = np.round(np.random.lognormal(2.3, 0.3, size=(n,)), decimals=2)
sersic = np.round(np.random.lognormal(0.5, 0.5, size=(n,)), decimals=2)

def make_gals(el, pa, re, sersic, size=(60, 60)):
    x,y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    gal_array = []
    for i in range(len(el)):
        mod = Sersic2D(amplitude=1, r_eff=re[i], n=sersic[i],
                       x_0=(size[0]-1)/2.0, y_0=(size[1]-1)/2.0,
                       ellip=el[i], theta=pa[i])
        img = mod(x, y)
        norm = img / np.max(img)
        gal_array.append(norm)
    return np.array(gal_array)

## reshape image and SED array to allow for multiplication
## combine sersic galaxies with input and target SEDs
def combine(gal_seds_in, gal_seds_out, el, pa, re, sersic):
    gal_images = make_gals(elip, PAs, Reff, sersic) # array of n galaxies
    gal_images = gal_images[...,None]
    gal_seds_in = gal_seds_in.T[:,None,None]
    gal_seds_out = gal_seds_out.T[:,None,None]
    gal_input = gal_images*gal_seds_in
    gal_target = gal_images*gal_seds_out
    return gal_input, gal_target

gal_input, gal_target = combine(gal_seds_in, gal_seds_out, elip, PAs, Reff, sersic)

## saving input and target galaxies to npy files
with open('inputgalaxies.npy','wb') as f:
    np.save(f,gal_input)
    np.save(f,z[z_in_idx])
    
with open('targetgalaxies.npy','wb') as g:
    np.save(g,gal_target)
    np.save(g,z[z_out_idx])