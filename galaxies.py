import sys
import h5py
import numpy as np
import argparse
from astropy.modeling.models import Sersic2D


## reshaping flux array into (redshifts,filters,fluxes)
def flux_reshape(flux):    
    a = flux.shape[0]
    b = flux.shape[1]
    flux0 = np.reshape(flux, (a, b, -1))
    return flux0


## random selection of SEDs
def choose_seds(flux0, n):
    # array of n SEDs
    sed_idx = np.random.choice(flux0.shape[2], size=n)
    gal_seds = flux0[:,:,sed_idx]
    return gal_seds


## selection for input and target galaxy redshift    
def input_target(gal_seds, z_idx=None):
    n = gal_seds.shape[2]
    if z_idx is None:
        max_z_idx = gal_seds.shape[0]
        z_in_idx = np.random.randint(1, max_z_idx - 1, size=n)
        z_out_idx = np.random.randint(z_in_idx + 1, max_z_idx, size=n)
    else:
        # use provided input and target redshift indices
        z_in_idx, z_out_idx = [i * np.ones(n) for i in z_idx]
    # get selected redshift for each SED
    gal_seds_in = gal_seds[z_in_idx, :, np.arange(n)]
    gal_seds_out = gal_seds[z_out_idx, :, np.arange(n)]
    return gal_seds_in, z_in_idx, gal_seds_out, z_out_idx


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
    gal_images = make_gals(el, pa, re, sersic) # array of n galaxies
    gal_images = gal_images[...,None]
    gal_seds_in = gal_seds_in[:,None,None]
    gal_seds_out = gal_seds_out[:,None,None]
    gal_input = gal_images * gal_seds_in
    gal_target = gal_images * gal_seds_out
    gal_input = gal_input.astype(np.float32)
    gal_target = gal_target.astype(np.float32)
    return gal_input, gal_target


def main(n=10000):
    ## HDF file produced by running candels_example.py
    filename = "candels.goodss.models.test.hdf"
    f = h5py.File(filename,'r')
    list(f.keys())

    z = np.array(f['z']) # redshift values
    wl = np.array(f['wl']) # wavelength for each filter
    flux = np.array(f['fluxes']) # flux at each redshift + filter
    flux *= 1e9
    flux = flux_reshape(flux)

    np.random.seed(4251)

    gal_seds = choose_seds(flux, n)
    
    gal_seds_in, z_in_idx, gal_seds_out, z_out_idx = input_target(gal_seds)

    ## generate sersic galaxies
    elip = np.round(np.random.uniform(low=0.0, high=0.8, size=(n,)), decimals=2)
    PAs = np.round(np.random.uniform(low=0.0, high=np.pi, size=(n,)), decimals=2)
    Reff = np.round(np.random.lognormal(2.3, 0.3, size=(n,)), decimals=2)
    sersic = np.round(np.random.lognormal(0.5, 0.5, size=(n,)), decimals=2)

    gal_input, gal_target = combine(gal_seds_in, gal_seds_out, elip, PAs, Reff, sersic)
    
    # reduce the number of filters included to 9
    gal_input = np.delete(gal_input, np.arange(1,17,2), axis = 3)
    gal_target = np.delete(gal_target, np.arange(1,17,2), axis = 3)

    # making input and target redshift arrays
    z_in = z[z_in_idx].astype(np.float32)
    z_out = z[z_out_idx].astype(np.float32)
    
    ## saving input and target galaxies to npy files
    np.save('inputgalaxies.npy', gal_input)
    np.save('inputredshifts.npy', z_in)
    np.save('targetgalaxies.npy', gal_target)
    np.save('targetredshifts.npy', z_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10000)
    args = parser.parse_args()
    sys.exit(main(n=args.n))
