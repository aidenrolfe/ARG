import sys
import h5py
import numpy as np
import argparse
from astropy.modeling.models import Sersic2D
from astropy.modeling.parameters import Parameter
from astropy.nddata import block_reduce
from scipy.special import gammaincinv, gammainc, gamma
from math import pi
from tqdm import tqdm


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


class Sersic2DAsym(Sersic2D):
    r"""
    Two dimensional Sersic profile with asymmetry.

    Parameters are same as Sersic2D, plus:
    ----------
    asym_strength : float, optional
        Strength of asymmetry.
    asym_angle : float, optional
        Position angle of maximum asymmetry.


    Notes
    -----
    Asymmetry introduced by multiplying Sersic2D profile by
    (1 - asym_strength * cosine(azimuthal angle - asym_angle)
    """

    asym_strength = Parameter(default=0)
    asym_angle = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta,
                 asym_strength, asym_angle):
        """Two dimensional Sersic profile function with asymmetry."""

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn = cls._gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        eps = 1e-32
        angle = np.arctan(x_maj/(x_min + eps))
        angle += np.pi * (x_min < 0) - np.pi/2
        angle[np.isnan(angle)] = 0
        asym = (1 - asym_strength * np.cos(theta - asym_angle - angle))
        return amplitude * asym * np.exp(-bn * (z ** (1 / n) - 1))


    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        par_unit = super()._parameter_units_for_data_units(inputs_unit, outputs_unit)
        return par_unit + {'asym_angle': u.rad}
    
## oversampling is not used in this case    
def oversamp_factor(Reff, sersic, elip, factor=25, minoversamp=5, maxoversamp=50, unbinned=False):
    oversamp = factor * sersic / (Reff * (1-elip))
    oversamp = np.minimum(oversamp, maxoversamp)
    oversamp = np.maximum(oversamp, minoversamp)
    if not unbinned:
        oversamp = np.ceil(oversamp).astype('int')
    return oversamp

def b(n):
    # Normalisation constant
    return gammaincinv(2*n, 0.5)

def sersic_lum(Ie, re, n):
    bn = b(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n

def sersic_enc_lum(r, Ie, re, n):
    x = b(n) * (r/re)**(1.0/n)
    return sersic_lum(Ie, re, n) * gammainc(2*n, x)

def make_gals(el, pa, re, sersic, asym, asym_angle, size=(60, 60),
              oversamp=None):
    if oversamp is None:
        oversamp = oversamp_factor(re, sersic, el)
    else:
        oversamp = oversamp * np.ones(len(re), dtype='int')

    amp = 1 / sersic_enc_lum(1.333, 1, re, sersic) / 0.195
    
    gal_array = []
    for i in tqdm(range(len(el))):
        oversamp = 1
        this_size = [x * oversamp for x in size]
        this_re = re[i] * oversamp
        mod = Sersic2DAsym(amplitude=amp[i], r_eff=this_re, n=sersic[i],
                       x_0=(this_size[0]-1)/2.0, y_0=(this_size[1]-1)/2.0,
                       ellip=el[i], theta=pa[i],
                       asym_strength=asym[i], asym_angle=asym_angle[i])
        x,y = np.meshgrid(np.arange(this_size[0]), np.arange(this_size[1]))
        img = mod(x, y)
        norm = img
        gal_array.append(norm)
    return np.array(gal_array)


## reshape image and SED array to allow for multiplication
## combine sersic galaxies with input and target SEDs
def combine(gal_seds_in, gal_seds_out, el, pa, re, sersic, asym, asym_angle):
    gal_images = make_gals(el, pa, re, sersic, asym, asym_angle) # array of n galaxies
    gal_images = gal_images[...,None]
    gal_seds_in = gal_seds_in[:,None,None]
    gal_seds_out = gal_seds_out[:,None,None]
    gal_input = gal_images * gal_seds_in
    gal_target = gal_images * gal_seds_out
    gal_input = gal_input.astype(np.float32)
    gal_target = gal_target.astype(np.float32)
    return gal_input, gal_target


def main(n=100):
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
    np.random.seed(1432)
    elip = np.round(np.random.uniform(low=0.0, high=0.8, size=(n,)), decimals=2)
    PAs = np.round(np.random.uniform(low=0.0, high=np.pi, size=(n,)), decimals=2)
    Reff = np.round(np.random.lognormal(2.3, 0.3, size=(n,)), decimals=2)
    sersic_low = np.round(np.random.lognormal(0.2, 0.2, size=(n//2,)), decimals=2)
    sersic_high = np.round(np.random.lognormal(1.3, 0.15, size=(n//2,)), decimals=2)
    sersic = np.concatenate((sersic_low, sersic_high))
    np.random.shuffle(sersic)
    asym_low = np.round(np.random.beta(0.5, 10, size=(n//2,)), decimals=2)
    asym_high = np.round(np.random.beta(10, 20, size=(n//2,)), decimals=2)
    asym = np.concatenate((asym_low, asym_high))
    np.random.shuffle(asym)
    asym_angle = np.round(np.random.uniform(low=0.0, high=2*np.pi, size=(n,)), decimals=2)

    gal_input, gal_target = combine(gal_seds_in, gal_seds_out, elip, PAs, Reff, sersic, asym, asym_angle)
    
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
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()
    sys.exit(main(n=args.n))
