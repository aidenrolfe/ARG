# ARG
Artificially Redshifting Galaxies with Neural Networks - Year 4 Project

## Instructions for generating galaxy images:

### Create a new conda environment with:

- Numpy
- SciPy
- Astropy
- H5PY
- Six

### Clone and install SMPY --> https://github.com/dunkenj/smpy

`git clone git://github.com/dunkenj/smpy.git`

`python setup.py install`

### Run the scripts

Run `candels_example.py` from the `scripts` folder of SMPY to generate the galaxy SED information used by `galaxies.py`.

Run `galaxies.py` from the `scripts` folder to generate `n` simulated galaxy images.

Run `redshifting.py` from the `scripts` folder to apply observational effects to the `n` galaxy images.

Run `Galaxies_VAE.py`from the `scripts` folder to train the network and produce image reconstructions, a 2D plot in latent space, and plot of model loss on training and validation sets.

### Note

The outputs will be `.npy` files.

`testplot.py` shows how the numpy arrays can be read and displayed.
