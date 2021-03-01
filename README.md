# ARG
Artificially Redshifting Galaxies with neural networks - Year 4 project

## Instructions for *devAiden*:

Installation of forked version of `SMPY` --> https://github.com/bamford/smpy


### Create a new conda environment with:
- Numpy
- SciPy
- Astropy
- H5PY
- Six


### Clone and install SMPY

`git clone git://github.com/bamford/smpy.git`

`python setup.py install`

### Run the scripts

Run `candels_example.py` from the `scripts` folder of SMPY to generate the needed HDF files

Run `plotting.py` from the `scripts` folder to generate `n` simulated galaxies.

### Note

The output will be a `.npy` file for both the input and target galaxies.

I would recommend reading the files in the following way:

```
with open('inputgalaxies.npy','rb') as f:

  gal_input = np.load(f)
  
  z_in = np.load(f)
```

```
with open('targetgalaxies.npy','rb') as g:

  gal_target = np.load(g)
  
  z_out = np.load(g)
```  
