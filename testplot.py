import matplotlib.pyplot as plt
import numpy as np 
import random

## reading galaxy images and redshifts 
with open('inputgalaxies.npy','rb') as f:
    gal_input = np.load(f)
    z_in_idx = np.load(f)
    
with open('targetgalaxies.npy','rb') as g:
    gal_target = np.load(g)
    z_out_idx = np.load(g)


m = gal_input.shape[0]
n = random.randint(0,m-1) # choosing a random galaxy to plot (as input and target redshift)

for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(gal_input[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
plt.suptitle('Redshift z = ' + str(np.round(z_in_idx,2)) + ' with galaxy image ' + str(n+1))
plt.show()

for i in range(17):
    plt.subplot(3,6,i+1)
    plt.imshow(gal_target[n,:,:,i], cmap='inferno',
               origin='lower', interpolation='nearest',
               vmin=0, vmax=1)
plt.suptitle('Redshift z = ' + str(np.round(z_out_idx,2)) + ' with galaxy image ' + str(n+1))
plt.show()