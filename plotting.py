from candels_example import models # an object created by the candels_example
import matplotlib.pyplot as plt
import numpy as np
import h5py

## attempting to read HDF file instead of importing the object
# models = h5py.File('test_hdf_set.hdf','r')

models.load_from_hdf
sed = models.sed_arr # SED arrays from models
neb_sed = models.neb_sed_arr # nebular SED array from models
sed2 = models.SED # multi dimensional SED array from models


# plotting contours of all 6 SED array
plt.contour(sed[0])
plt.contour(sed[1])
plt.contour(sed[2])
plt.contour(sed[3])
plt.contour(sed[4])
plt.contour(sed[5])
plt.show()


# plotting contours of all 6 nebular SED array
plt.contour(neb_sed[0])
plt.contour(neb_sed[1])
plt.contour(neb_sed[2])
plt.contour(neb_sed[3])
plt.contour(neb_sed[4])
plt.contour(neb_sed[5])
plt.show()


#plotting an array from both SED and nebular SED as an image
plt.imshow(sed[0], origin='lower', interpolation='nearest',
           vmin=-1, vmax=2)
plt.show()

plt.imshow(neb_sed[0], origin='lower', interpolation='nearest',
           vmin=-1, vmax=2)
plt.show()