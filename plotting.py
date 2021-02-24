import h5py
import numpy as np
import matplotlib.pyplot as plt 

# HDF file produced by running candels_example.py from SMPY
filename = "candels.goodss.models.test.hdf"

f = h5py.File(filename,'r')

list(f.keys())

z = np.array(f['z']) # redshift values

mag = np.array(f['mags']) # magnitude at each redshift + filter

## reshaping magnitude array into (x,y,z)
# x = no. of redshifts
# y = no. of filters
# z = no. of magnitude values at each filter
mag0 = np.reshape(mag,(11,17,528))

flux = np.array(f['fluxes']) # flux at each redshift + filter

flux0 = np.reshape(flux,(11,17,528))

wl = np.array(f['wl']) # wavelength for each filter

meanflux = flux0.mean(axis=2) # average the fluxes for each filter

a = meanflux.shape[0]


for i in range(a):
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Flux SED at redshift ' + str(z[i]))
    plt.plot(wl,meanflux[i])
    plt.show()
    

meanmag = mag0.mean(axis=2) # average the magnitudes for each filter

b = meanmag.shape[0]

for j in range(b):
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude SED at redshift ' + str(z[j]))
    plt.plot(wl,meanmag[j])
    plt.show()
    